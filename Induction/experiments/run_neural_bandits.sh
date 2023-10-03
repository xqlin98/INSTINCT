datasets=(active_to_passive antonyms auto_categorization auto_debugging cause_and_effect common_concept diff first_word_letter informal_to_formal larger_animal letters_list negation num_to_verbal odd_one_out object_counting orthography_starts_with periodic_elements rhymes second_word_letter sentence_similarity sentiment singular_to_plural sum synonyms taxonomy_animal translation_en-de translation_en-es translation_en-fr word_sorting word_unscrambling)
for i in ${datasets[@]}; do
    python experiments/run_neural_bandits.py \
    --task $i \
    --n_prompt_tokens 5 \
    --nu 1 \
    --lamdba 0.1 \
    --n_init 40 \
    --n_domain 10000 \
    --total_iter 165 \
    --local_training_iter 1000 \
    --n_eval 1000 \
    --intrinsic_dim 10 \
    --gpt gpt-3.5-turbo-0301 \
    --name iter165_gpt-0301
done