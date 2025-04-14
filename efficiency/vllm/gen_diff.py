import json

file_ngram = (
    "results/DeepSeek-R1-Distill-Llama-8B_ngram-8_aimo-validation-aime_32_0.6.jsonl"
)
file_full = (
    "results/DeepSeek-R1-Distill-Llama-8B_full_aimo-validation-aime_32_0.6.jsonl"
)

with open(file_ngram) as f_ngram:
    ngram_lines = f_ngram.readlines()

with open(file_full) as f_full:
    full_lines = f_full.readlines()

ngram_data = json.loads(ngram_lines[0])
full_data = json.loads(full_lines[0])

diffs = []

for id, ngram_gen in enumerate(ngram_data["generated_texts"]):
    print("Problem ID: ", id)
    print(ngram_gen[:20])
    full_gen = full_data["generated_texts"][id]
    print(full_gen[:20])

    diff = ngram_gen != full_gen
    print("Diff:", diff)
    diffs.append(diff)

    if diff:
        print(
            "Len eqaul: ",
            len(ngram_gen) == len(full_gen),
            len(ngram_gen),
            len(full_gen),
        )
        for i in range(min(len(ngram_gen), len(full_gen))):
            if ngram_gen[i] != full_gen[i]:
                print(
                    f"Diff at index {i}: Ngram: <{ngram_gen[max(0, i-10):i+30]}> != Full: <{full_gen[max(0, i-10):i+30]}>"
                )
                break

print("All diff:", any(diffs))
