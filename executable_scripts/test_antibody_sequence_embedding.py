import numpy as np
import pandas as pd
import argparse
from scipy.stats import gamma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_subj', help='', default=80)
    parser.add_argument('--subj_sample_size', help='', default=100)
    parser.add_argument('--embedding_dim', help='', default=100)
    parser.add_argument('--output_file', help='test.tsv')
    execute(args)


def execute(args):
    num_of_subj = args.num_of_subj
    subj_sample_size = args.subj_sample_size
    embedding_dim = args.embedding_dim
    output_file = args.output_file

    np.random.seed(0)

    subjects = pd.DataFrame(index=range(num_of_subj),
                            columns=('repertoire.diseases_diagnosis', 'repertoire.repertoire_name'))
    subjects['repertoire.diseases_diagnosis'] = 'Healthy'
    subjects['repertoire.repertoire_name'] = range(num_of_subj)
    subjects.loc[np.random.choice(list(range(num_of_subj)), size=int(num_of_subj/2), replace=False),
                 'repertoire.diseases_diagnosis'] = 'Sick'

    samples = pd.DataFrame()
    for idx, subj in subjects.iterrows():
        num_of_samples = int(subj_sample_size * (1 + np.random.sample()) / 2) * 2
        tmp_index = np.array(range(num_of_samples)) + 10000*idx
        tmp = pd.DataFrame(index=tmp_index, columns=['embedding'])
        if subj['repertoire.diseases_diagnosis'] == 'Healthy':
            data1 = gamma(1).rvs((int(num_of_samples/2), embedding_dim), random_state=0)
            data2 = gamma(2).rvs((int(num_of_samples/2), embedding_dim), random_state=0)
            data = np.concatenate([data1, data2])
        else:
            data1 = gamma(1).rvs((int(num_of_samples/2), embedding_dim), random_state=0)
            data2 = gamma(3).rvs((int(num_of_samples/2), embedding_dim), random_state=0)
            data = np.concatenate([data1, data2])
        tmp.loc[tmp_index, 'embedding'] = [x for x in data]
        tmp.loc[tmp_index, 'repertoire.diseases_diagnosis'] = subj['repertoire.diseases_diagnosis']
        tmp.loc[tmp_index, 'repertoire.repertoire_name'] = int(subj['repertoire.repertoire_name'])
        samples = pd.concat([samples, tmp])

    samples.to_csv(output_file, sep='\t')
    print('test samples file saved to {}'.format(output_file))


if __name__ == "__main__":
    main()
