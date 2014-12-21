#!/usr/bin/env python
#
# Copyright (c) 2014 In-Q-Tel, Inc/Lab41, All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import glob
import json
import sys
import numpy as np
from math import sqrt
import scipy.sparse
import scipy.spatial.distance
import matplotlib.pyplot as plt

import random
from sklearn.naive_bayes import GaussianNB

# Threshold above which two communities are considered the "same"
COSINE_THRESHOLD = .85

def analyze_metrics(dataset, output_dir, file_names, metrics_to_evaluate):
    """
        Creates histograms of specific metrics across algorithms

        Args:
           dataset (string): dataset being processed [used for naming output file]
           output_dir (string): output path
           file_names (list of strings): Input metrics json files
           metrics_to_evaluate (list of strings): Metrics to be histogramed
        Return:
            None
    """
    num_files = len(file_names)
    # Load metrics into memory
    results = []
    for json_path in file_names:
        print(json_path)
        with open(json_path) as f:
            results.append(json.load(f))

    # Numpy matrix for each dataset
    matricies = []
    num_communities = []
    for result in results:
        # Iterate through results getting # of communities by number of nodes
        num_nodes = len(result['membership'])
        max_seen = -1
        for node in result['membership']:
            if len(node) > 0:
                max_val = max(node)
                if max_val > max_seen:
                    max_seen = max_val
            else:
                print('ERROR:', node)

        print('Max Community: ', max_seen)
        max_seen +=1 # Handle that communities are zero indexed
        num_communities.append(max_seen)
        # Create matrix of nodes (rows) vs community #
        community_matrix = scipy.sparse.csc_matrix((num_nodes, max_seen))
        for i, node in enumerate(result['membership']):
            for community in node:
                community_matrix[i, community] = 1

        print('Shape: ', community_matrix.shape)
        matricies.append(community_matrix)

    total_communities = 0
    for max_seen in num_communities:
        total_communities += max_seen

    community_n2 = scipy.sparse.dok_matrix((total_communities,total_communities))
    metric_names = [metric_name for metric_name in results[0]['metrics'].keys()]

    found = [ [] for x in range(len(metric_names))]
    notFound = [[] for x in range(len(metric_names))]

    # Iterate through sets of results (comparing leftMatrix to rightMatrix)
    for i, leftMatrix in enumerate(matricies):
        if i != 0:
            i_base += num_communities[i-1]
        else:
            i_base = 0
        for j, rightMatrix in enumerate(matricies[i+1:]):
            #print('i, j: (%d, %d)'%(i, i+1+j),leftMatrix.shape, rightMatrix.shape)
            print(os.path.basename(file_names[i]), os.path.basename(file_names[i+1+j]))

            # Compare each community in the left results set to each community in the right results set
            # Comparing leftCol from leftMatrix to rightCol from rightMatrix
            j_base = sum(num_communities[:i+1+j])
            for leftCol  in range(leftMatrix.shape[1]):
                max_cos = -1
                right_index = -1
                for rightCol in range(rightMatrix.shape[1]):
                    l = leftMatrix.getcol(leftCol)
                    r = rightMatrix.getcol(rightCol)

                    # Cosine similarity [ended up implementing due to trouble with the numpy distance metric]
                    dot = l.transpose().dot(r)[0,0]
                    norm = sqrt(l.transpose().dot(l)[0,0])* sqrt(r.transpose().dot(r)[0,0])
                    distance = dot/norm

                    # Keep track of "closest" vector
                    if distance > max_cos:
                        rightIndex = rightCol
                        max_cos = distance

                # If closest vector is close enough to consider it to be the same cluster
                if max_cos > COSINE_THRESHOLD:
                    # Add to found count for that metric
                    for k, metric_name in enumerate(metric_names):
                        found[k].append(results[i]['metrics'][metric_name]['results'][leftCol])
                    #print (i_base + leftCol, j_base + rightIndex)
                    community_n2[i_base + leftCol, j_base + rightIndex] = max_cos
                    community_n2[j_base + rightIndex, i_base + leftCol] = max_cos
                else:
                     # Add to found count for that metric
                    for k, metric_name in enumerate(metric_names):
                        notFound[k].append(results[i]['metrics'][metric_name]['results'][leftCol])


    print('Found:', len(found[0]))
    print('Not Found:', len(notFound[0]))

    # Run through naive-bayes classifier?
    # Add found and not found to same matrix
    data = np.concatenate((np.array(found).transpose(), np.array(notFound).transpose()), 0)
    target = np.concatenate((np.ones(len(found[0])), np.zeros(len(notFound[0]))), 0)
    print('F, NF', len(found[0]), len(notFound[0]))

    HOLDOUT_PERCENT= 0.1
    # TODO: This should implement some sort of sampling
    allIDX = np.arange(data.shape[0])
    random.shuffle(allIDX)
    holdout_count = int(data.shape[0]*HOLDOUT_PERCENT)
    test_IDX = allIDX[:holdout_count]
    train_IDX = allIDX[holdout_count:]

    print(train_IDX.shape, data.shape)
    data_train = data[train_IDX, :]
    target_train = target[train_IDX, :]
    data_test = data[test_IDX, :]
    target_test = target[test_IDX, :]

    gnb = GaussianNB()
    #gnb.fit(data_train, target_train)
    #print('Classification Accuracy: ', gnb.)
    y_pred = gnb.fit(data_train, target_train).predict(data_test)
    print("Number of mislabeled points out of a total %d points : %d" % (data_test.shape[0],(target_test != y_pred).sum()))


    # # Plot Histograms
    # bins = np.linspace(0, 1, num=20)
    # font_size = 8
    # font = {'size': font_size}
    # offset = 0
    # for offset in [0, 16]:
    #     # Iterate through subfigures
    #     plt.clf()
    #     for i in range(offset, 16+offset):
    #         plt.rc('font', **font)
    #         plt.subplot(4,4, 1+ i-offset) # Subplots are 1 indexed
    #
    #         min_val = min(min(found[i]), min(notFound[i]))
    #         max_val = max(max(found[i]), max(notFound[i]))
    #         bins = np.linspace(min_val, max_val, num=10)
    #         plt.hist(found[i], bins, alpha=0.5, label='f')
    #         plt.hist(notFound[i], bins, alpha=0.5, label='nf')
    #         plt.legend(loc='upper right', fontsize=font_size)
    #         # Break long names into two lines
    #         # TODO: Improve to split on space
    #         if len(metric_names[i]) < 20:
    #             plt.title(metric_names[i], fontsize=font_size)
    #         else:
    #             name = metric_names[i][:20] + '\n' + metric_names[i][20:]
    #             plt.title(name, fontsize=font_size)
    #     fig = plt.gcf()
    #     fig.tight_layout()
    #     plt.savefig('test%d.png'%offset)
    #
    # # Original experiment shwoing how each community compares to every other community
    # output_path = os.path.join(output_dir, 'community_comparison.csv')
    # print('Saving to: %s'%output_path)
    # header_row = []
    # print('Len Num Comm: ', len(num_communities))
    # for i, community_size in enumerate(num_communities):
    #     algorithm = file_names[i].split('--')[1]
    #     print(i, algorithm)
    #     for j in range(community_size):
    #         val = '%d - %s'%(j, algorithm)
    #         header_row.append(val)
    #
    # np.savetxt(output_path, community_n2.todense(), delimiter=",", header=','.join(header_row))
    print('Done')


def main():
    parser = argparse.ArgumentParser(description=
                                     'Create side by side histograms for various metrics across algorithms for a given dataset')
    parser.add_argument('input_path', type=str, help='file or directory containing  metric json files')
    parser.add_argument('dataset', type=str, help='Dataset desired (i.e. football)')
    parser.add_argument('--metrics', type=str,
                        default=','.join(['Separability', 'Cohesiveness', 'Density', 'Triangle Participation Ratio', 'Conductance']),
                        help='Metrics to Compare (comma separated)')
    parser.add_argument('--output', type=str, default=os.getcwd(), help='Base output directory')
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print("Path \"{}\" does not exist".format(args.input_path))
        return


    if os.path.isdir(args.input_path):
        file_names = glob.glob(os.path.join(args.input_path, '*%s*.json'%args.dataset))
        print(file_names)
        files_to_process = []
        for file_name in file_names:
            if 'info'  in file_name or 'ground' in file_name:
                files_to_process.append(file_name)
        analyze_metrics(args.dataset, args.output, files_to_process, args.metrics.split(','))
    else:
        analyze_metrics(args.dataset, args.output, [args.input_path], args.metrics.split(','))

if __name__ == "__main__":
    main()
