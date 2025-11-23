import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from tah.model.recurrent_transformer import TaHForCausalLM
from tah.model.tracker import TaHTracker
from tah.model.utils import IterCountColors, TaHForCasualLM_generate

# Fix random seed for reproducibility
torch.manual_seed(42)

def main():
    """
    Initializations
    """

    save_model_name = "nics-efc/TaH-plus-1.7B"
    device_map = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(save_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    override_config = None
    
    tah_model = TaHForCausalLM.from_pretrained(
        save_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
        tah_config=override_config,
    )

    device = tah_model.device
    dtype = tah_model.dtype
    print(f"Device: {device}, Dtype: {dtype}")

    tah_model = tah_model.to(dtype=dtype)

    # Attach tracker
    tracker = TaHTracker(top_k=10)
    tracker.attach(tah_model)

    """
    Input and run
    """

    # prepare the model input
    prompts = [
        "Six points $A, B, C, D, E$ and $F$ lie in a straight line in that order. Suppose that $G$ is a point not on the line and that $AC = 26$, $BD = 22$, $CE = 31$, $DF = 33$, $AF = 73$, $CG = 40$, and $DG = 30$. Find the area of $\\triangle BGE$."
    ]

    # Process each prompt through chat template
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
        )
        texts.append(text)

    model_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, padding_side="left"
    ).to(device=device)
    batch_size = model_inputs.input_ids.shape[0]

    print("Initial input:")
    for i in range(batch_size):
        print(f"\nSample {i+1}:")
        print(f"Prompt: {prompts[i][:100]}{'...' if len(prompts[i]) > 100 else ''}")
    print(f"Input IDs shape: {model_inputs.input_ids.shape}")

    print(IterCountColors.get_legend())

    # Use the generation function with sampling
    output_tokens, final_texts = TaHForCasualLM_generate(
        tah_model=tah_model,
        tokenizer=tokenizer,
        model_inputs=model_inputs,
        iter_count=None,  # Use automatic iteration from iter_decider
        max_new_tokens=16384,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        verbose=True
    )

    # analyze the token count for batch
    batch_size = model_inputs.input_ids.shape[0]
    max_input_length = model_inputs.input_ids.shape[1]

    print("\n" + "=" * 50)
    print("TOKEN COUNT ANALYSIS")
    print("=" * 50)
    print(f"Batch size: {batch_size}")
    print(f"Max input length (with padding): {max_input_length}")

    # Calculate actual input lengths (excluding padding)
    actual_input_lengths = []
    for i in range(batch_size):
        actual_length = (
            (model_inputs.input_ids[i] != tokenizer.pad_token_id).sum().item()
        )
        actual_input_lengths.append(actual_length)
        print(f"Sample {i+1} actual input length: {actual_length}")

    # For generated tokens, we now get a list of lists for each batch item
    generated_counts = [len(seq) for seq in output_tokens]
    total_generated = sum(generated_counts)
    print(f"Generated tokens per sample: {generated_counts}")
    print(f"Total generated tokens: {total_generated}")
    print("=" * 50)

    # Print final generated texts for each sample in the batch
    print("\nFINAL GENERATED TEXTS:")
    print("=" * 50)
    for i, text in enumerate(final_texts):
        print(f"\nSample {i+1} output:")
        print("-" * 30)
        print(text)

    # Display recorded information from tracker
    record_pd = tracker.to_pandas()
    print(record_pd)

    # Save tracker records to CSV
    if not os.path.exists("output/analysis"):
        os.makedirs("output/analysis")
    record_pd.to_csv("output/analysis/tracker_records.csv", index=False)

    tracker.detach()

    print(f"\nBatch inference completed successfully for {batch_size} samples!")

if __name__ == "__main__":
    main()
