

test_id_list = []
for i in idx:
    random_idx = np.random.choice([j for j in idx if j != i and j != pairs - 1], 1, replace=True)
    test_id_list.append(str(random_idx[0]))
save_list(os.path.join(LISTS_DIRECTORY, "dummy_id.txt"), test_id_list)