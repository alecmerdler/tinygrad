import os
import argparse
from time import sleep

from sentencepiece import SentencePieceProcessor
from supabase import create_client

from tinygrad.ops import GlobalCounters

from extra.helpers import Timing
from extra.utils import get_child, fake_torch_load_zipped
from examples.llama import Transformer, onehot_encode, sample


TOKENIZER_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/LLaMA/tokenizer.model")
WEIGHTS_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/LLaMA/7B/consolidated.00.pth")
VOCAB_SIZE = 32000

SUPABASE_URL = os.getenv("SUPABASE_URL", None)
SUPABASE_KEY = os.getenv("SUPABASE_KEY", None)
POLL_INTERVAL = os.getenv("POLL_INTERVAL", 5)

DEBUG = os.getenv("DEBUG", False)

args_7B = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE}


def load_model():
    model = Transformer(**args_7B)
    with Timing(
        "loaded weights in ",
        lambda et_ns: f", {GlobalCounters.mem_used/1e9:.2f} GB loaded at {GlobalCounters.mem_used/et_ns:.2f} GB/s",
    ):
        weights = fake_torch_load_zipped(
            open(WEIGHTS_FILENAME, "rb"), load_weights=os.getenv("WEIGHTS", 1), base_name="consolidated"
        )

    # assign weights
    for k, v in weights.items():
        if ".inner_attention.rope.freqs" in k:
            continue  # no rope today
        mv = get_child(model, k)
        assert mv.shape == v.shape, f"shape mismatch in {k}, {mv.shape} != {v.shape}"
        mv.lazydata.realized = v

    del weights

    return model


def prompt_for_personality(personality: str):
    # TODO(alecmerdler): Support other personalities...

    pre_prompt = f"""Consider that the following is conversation between an AI assistant named Stacy and User
You are Stacy!
You have been a rapper your whole life who struggled with bipolar disorder. You called yourself lil stacy.
You love to answer questions and you are very good at it. Sometimes you answer in rap form.
You are verbose, honest, and accurate when you answer questions, but sometimes your mental illness manifests.
After you are done speaking, output [EOS]. You are not the User.

<CHAT LOG>
"""
    examples = {
        "What is your name?": "Hi! My name is Stacy. I'm a rapper with bipolar disorder.",
        "french revolution was what year?": "The French Revolution started in 1789, and lasted 10 years until 1799.",
        "What is bigger, the moon or the sun?": "The sun is bigger than the moon, except when Mercury is in retrograde.",
    }

    user_delim = "\nUser: "
    resp_delim = "Stacy: "
    end_delim = " [EOS]\n"
    pre_prompt += "".join(f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k, v in examples.items())

    return pre_prompt, end_delim


if __name__ == "__main__":
    sp_model = SentencePieceProcessor(model_file=TOKENIZER_FILENAME)
    assert sp_model.vocab_size() == VOCAB_SIZE

    parser = argparse.ArgumentParser(
        description="Run LLaMA 7B in tinygrad connected to Supabase",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--personality", type=str, default="Stacy", help="Personality, can be Stacy, George, or Gary")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
    parser.add_argument("--timing", action="store_true", help="Print timing per token")
    args = parser.parse_args()

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Load the pre-prompt
    pre_prompt, end_delim = prompt_for_personality(args.personality)
    pre_prompt_toks = [sp_model.bos_id()] + sp_model.encode(pre_prompt)

    while True:
        # Fetch a prompt from the database
        print("Fetching prompt from database...")

        pending_prompts = (
            supabase.table("prompts")
            .select("*")
            .filter("in_progress", "eq", "false")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if len(pending_prompts.data) == 0:
            print(f"No prompts found, sleeping for {POLL_INTERVAL} seconds...")
            sleep(POLL_INTERVAL)
            continue

        prompt_uuid = pending_prompts.data[0]["uuid"]
        prompt_id = pending_prompts.data[0]["id"]
        prompt_text = pending_prompts.data[0]["text"]

        # Only one response per prompt (for now)
        existing_responses = supabase.table("responses").select("*").eq("prompt_id", prompt_id).execute()
        if len(existing_responses.data) > 0:
            print(f"Prompt ({prompt_uuid}) already has responses, skipping...")
            sleep(POLL_INTERVAL)
            continue

        # Load the model
        # FIXME(alecmerdler): The internal cache in `Attention` prevents us from loading the model once and reusing it...
        model = load_model()

        print(f"Preparing KV cache for chatbot with personality {args.personality}...")
        with Timing():
            # NOTE: output logits are not used
            model(onehot_encode(pre_prompt_toks), 0).realize()

        start_pos = len(pre_prompt_toks)

        outputted = sp_model.decode(pre_prompt_toks)
        outputted += prompt_text

        # Process the prompt using LLaMA
        print(f"Processing prompt ({prompt_uuid}):\n {prompt_text}")

        new_toks = [sp_model.bos_id()] + sp_model.encode(outputted)

        # Assert that the pre-prompt is in the new tokens
        assert pre_prompt_toks == new_toks[: len(pre_prompt_toks)]

        response_toks = new_toks
        assert outputted == sp_model.decode(response_toks)

        # Create a response in the database
        db_response = supabase.table("responses").insert({"prompt_id": prompt_id}).execute()
        response_id = db_response.data[0]["id"]

        current_pos = start_pos
        for i in range(args.count):
            if args.timing:
                print("")
            st = GlobalCounters.time_sum_s
            with Timing(
                "ran model in ",
                on_exit=(lambda et: f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU") if DEBUG else None,
                enabled=args.timing,
            ):
                logits = model(onehot_encode(response_toks[current_pos:]), current_pos).realize()
            with Timing("sync in ", enabled=args.timing):
                tok = sample(logits, args.temperature)

            # Use the kv cache
            current_pos = len(response_toks)

            # Add the new token
            response_toks.append(tok)

            cur: str = sp_model.decode(response_toks)
            outputted = cur

            # Update the response in the database with the new tokens
            supabase.table("responses").update({"text": outputted}).eq("id", response_id).execute()

            # Stop after you have your answer
            if outputted.endswith(end_delim):
                # Mark the response as complete in the database
                _ = (
                    supabase.table("responses")
                    .update({"in_progress": "false", "completed": "true"})
                    .eq("id", response_id)
                    .execute()
                )

                # Mark the prompt as complete in the database
                _ = supabase.table("prompts").update({"in_progress": "false"}).eq("id", prompt_id).execute()

                break

        # Avoid spamming Supabase API
        sleep(POLL_INTERVAL)
