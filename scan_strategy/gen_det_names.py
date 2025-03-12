
import random
import pickle

def generate_strings(n=64):
    strings = set()
    while len(strings) < n:
        rand_digits = f"{random.randint(0, 999):03d}"
        strings.add(f"A_{rand_digits}")
    return list(strings)

if __name__ == "__main__":

    d = {}
    d['det_name_HF'] = generate_strings()
    pickle.dump(d, open('dicts_dir/TIM_det_names.p', 'wb'))