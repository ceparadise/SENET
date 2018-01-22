import random

with open('./RNN_result119.txt', encoding='utf8') as fin, open('./pilot_res.csv', 'w', encoding='utf8') as fout:
    golden_set = set()
    predict_set = set()
    for line in fin:
        parts = line.split('\t')
        if parts[0] != 'Yes' and parts[0] != 'No':
            continue
        golden_label = parts[0]
        correctness = parts[1]
        w1 = parts[2]
        w2 = parts[4]
        if golden_label == 'Yes':
            if correctness == 'Correct':
                predict_res = 'pred_yes'
            else:
                predict_res = 'pred_no'
            golden_set.add((w1, w2, 'gold_yes', predict_res))
        if golden_label == 'No' and correctness == 'Incorrect':
            predict_set.add((w1, w2, 'golen_no', 'pred_yes'))

    merged_list = random.sample(golden_set, 50)
    merged_list.extend(random.sample(predict_set, 50))
    random.shuffle(merged_list)

    for pair in merged_list:
        fout.write("{},{},{},{}\n".format(pair[0], pair[1], pair[2], pair[3]))
