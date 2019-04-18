import os
import math
import copy

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from regression.data_ops import normalize, numerical, transpose


def polynomial(x, coeffs):
    y = 0
    for i in range(len(coeffs)):
        y += coeffs[i] * (x ** i)
    return y


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def extract_features(dataset, missed, show=False):
    result_for_event = []

    ts_start = float(dataset[0][0])
    ts_end = float(dataset[0][-1])

    result_for_event.append(ts_start)
    result_for_event.append(ts_end - ts_start)

    eda_sum = 0.0
    eda_cnt = 0

    for e in dataset[5]:
        if e != '':
            eda_sum += float(e)
            eda_cnt += 1

    avg_eda = eda_sum / eda_cnt
    result_for_event.append(avg_eda)

    temp_sum = 0.0
    temp_cnt = 0

    for t in dataset[7]:
        if t != '':
            temp_sum += float(t)
            temp_cnt += 1

    avg_temp = temp_sum / temp_cnt
    result_for_event.append(avg_temp)

    fidget_list = []

    for a_x, a_y, a_z in zip(dataset[1], dataset[2], dataset[3]):
        if a_x != '' and a_y != '' and a_z != '':
            fidget_list.append(math.sqrt(float(a_x) ** 2 + float(a_y) ** 2 + float(a_z) ** 2))

    avg_fidget = np.std(fidget_list)
    result_for_event.append(avg_fidget)

    min_bvp = float(min([float(d) for d in dataset[4] if d != '']))
    max_bvp = float(max([float(d) for d in dataset[4] if d != '']))

    result_for_event.append(min_bvp)
    result_for_event.append(max_bvp)

    # run regression
    bvp_data = [dataset[0], dataset[4]]
    numerical(bvp_data)
    normalize(bvp_data)

    # create matrices
    x_data = []
    y_data = []
    for idx in range(len(bvp_data[0])):
        row = []

        for d in range(12):
            row.append(bvp_data[0][idx] ** d)

        y_data.append([bvp_data[1][idx]])
        x_data.append(row)

    x_arr = np.array(x_data)
    y_arr = np.array(y_data)

    # apply some regularization
    reg_arr = .0001 * np.identity(len(x_data[0]))
    reg_arr[0, 0] = 0

    # and regress
    theta_arr = np.dot(np.matrix.transpose(x_arr), x_arr)
    theta_arr = theta_arr + reg_arr
    theta_arr = np.linalg.inv(theta_arr)
    theta_arr = np.dot(theta_arr, np.matrix.transpose(x_arr))
    theta_arr = np.dot(theta_arr, y_arr)

    result_for_event.extend([theta[0] for theta in theta_arr.tolist()])

    error = 0
    for i in range(len(x_data)):
        hyp = 0

        for j in range(len(x_data[i])):
            hyp += x_data[i][j] * theta_arr[j]

        error += (hyp - y_data[i][0]) ** 2

    error /= len(dataset[0])
    result_for_event.extend(error.tolist())
    result_for_event.append(missed)

    if show:
        print(theta_arr)
        line_x = np.linspace(0, 1000)

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)

        ax.set_xlabel('ts')
        ax.set_ylabel('bvp')
        ax.set_title(error)
        ax.scatter(bvp_data[0], bvp_data[1])
        ax.plot(line_x, polynomial(line_x, theta_arr))

        plt.show()

    return result_for_event


def get_coeffs_from_data(bvp_data, reg, degree):
    x_data = []
    y_data = []

    for idx in range(len(bvp_data[0])):
        row = []

        for d in range(degree):
            row.append(bvp_data[0][idx] ** d)

        y_data.append([bvp_data[1][idx]])
        x_data.append(row)

    x_arr = np.array(x_data)
    y_arr = np.array(y_data)

    # apply some regularization
    reg_arr = reg * np.identity(len(x_data[0]))
    reg_arr[0, 0] = 0

    # and regress
    theta_arr = np.dot(np.matrix.transpose(x_arr), x_arr)
    theta_arr = theta_arr + reg_arr
    theta_arr = np.linalg.inv(theta_arr)
    theta_arr = np.dot(theta_arr, np.matrix.transpose(x_arr))
    theta_arr = np.dot(theta_arr, y_arr)

    return [i[0] for i in list(theta_arr.tolist())]


def vis_boundary_set(range_list, coeff_list, bvp_list, valid_list, reg_errors, coeff_errors, size_errors, min_idx):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(30, 30)
    axes[0].set_title('neighboring coefficients for boundary set')

    scatter_x = [i for i in range(range_list[0][0], range_list[-1][1])]
    coeff_scatter_y_list = [[] for _ in range(len(coeff_list[0]))]

    for b_range, b_coeff_list in zip(range_list, coeff_list):
        for idx, b in enumerate(b_coeff_list):
            coeff_scatter_y_list[idx].extend([b for _ in range(b_range[1] - b_range[0])])

    coeff_scatter_colors = cm.rainbow(np.linspace(0, 1, len(coeff_scatter_y_list)))

    for coeff_scatter_y, color in zip(coeff_scatter_y_list, coeff_scatter_colors):
        axes[0].scatter(scatter_x, coeff_scatter_y, c=color)

    axes[1].set_title('bvp pulses for boundary set')

    bvp_scatter_y = []

    for idx in range(len(bvp_list)):
        bvp_scatter_y.extend(bvp_list[idx][1])

    axes[1].scatter(scatter_x, bvp_scatter_y)

    for r, v in zip(range_list, valid_list):
        if v == 1:
            axes[1].axvspan(r[0], r[1], color='green', alpha=.25)
        elif v == -1:
            axes[1].axvspan(r[0], r[1], color='red', alpha=.25)
        else:
            axes[1].axvspan(r[0], r[1], color='blue', alpha=.25)

    # axes[2].set_title('error landscape for current boundary')
    # error_scatter_x = [i for i in range(len(reg_errors))]
    #
    # axes[2].axvline(min_idx)
    # axes[2].axvline([idx for idx in range(len(reg_errors)) if reg_errors[idx] == min(reg_errors)][0], color='green')
    # axes[2].axvline([idx for idx in range(len(coeff_errors)) if coeff_errors[idx] == min(coeff_errors)][0], color='blue')
    # axes[2].axvline([idx for idx in range(len(size_errors)) if size_errors[idx] == min(size_errors)][0], color='red')
    #
    # normalize([reg_errors, coeff_errors, size_errors])
    #
    # sums = [r + c + s for r, c, s in zip(reg_errors, coeff_errors, size_errors)]
    #
    # axes[2].scatter(error_scatter_x, reg_errors, color='green')
    # axes[2].scatter(error_scatter_x, coeff_errors, color='blue')
    # axes[2].scatter(error_scatter_x, size_errors, color='red')
    # axes[2].scatter(error_scatter_x, sums, color='purple')

    plt.show()


def split_boundary_low(range_list, coeff_list, bvp_list, valid_list, b_pos):
    range_list.insert(b_pos, (range_list[b_pos][0], range_list[b_pos][0] + 1))
    range_list[b_pos + 1] = (range_list[b_pos + 1][0] + 1, range_list[b_pos + 1][1])

    bvp_list.insert(b_pos, [[bvp_list[b_pos][0][0]], [bvp_list[b_pos][1][0]]])
    del bvp_list[b_pos + 1][0][0]
    del bvp_list[b_pos + 1][1][0]

    valid_list.insert(b_pos, 0)

    reg_data1 = copy.deepcopy(bvp_list[b_pos])
    normalize(reg_data1)
    coeff_list.insert(b_pos, get_coeffs_from_data(reg_data1, regularization, degree))

    reg_data2 = copy.deepcopy(bvp_list[b_pos + 1])
    normalize(reg_data2)
    coeff_list[b_pos + 1] = get_coeffs_from_data(reg_data2, regularization, degree)


def split_boundary_high(range_list, coeff_list, bvp_list, valid_list, b_pos):
    range_list.insert(b_pos + 1, (range_list[b_pos][1] - 1, range_list[b_pos][1]))
    range_list[b_pos] = (range_list[b_pos][0], range_list[b_pos][1] - 1)

    bvp_list.insert(b_pos + 1, [[bvp_list[b_pos][0][-1]], [bvp_list[b_pos][1][-1]]])
    del bvp_list[b_pos][0][-1]
    del bvp_list[b_pos][1][-1]

    valid_list.insert(b_pos + 1, 0)

    reg_data1 = copy.deepcopy(bvp_list[b_pos])
    normalize(reg_data1)
    coeff_list[b_pos] = get_coeffs_from_data(reg_data1, regularization, degree)

    reg_data2 = copy.deepcopy(bvp_list[b_pos + 1])
    normalize(reg_data2)
    coeff_list.insert(b_pos + 1, get_coeffs_from_data(reg_data2, regularization, degree))


def delete_boundary_merge_down(range_list, coeff_list, bvp_list, valid_list, b_pos):
    range_list[b_pos - 1] = (range_list[b_pos - 1][0], range_list[b_pos][1])
    del range_list[b_pos]

    bvp_list[b_pos - 1][0].extend(bvp_list[b_pos][0])
    bvp_list[b_pos - 1][1].extend(bvp_list[b_pos][1])

    del bvp_list[b_pos]
    del valid_list[b_pos]
    del coeff_list[b_pos]

    reg_data1 = copy.deepcopy(bvp_list[b_pos - 1])
    normalize(reg_data1)
    coeff_list[b_pos - 1] = get_coeffs_from_data(reg_data1, regularization, degree)


def delete_boundary_merge_up(range_list, coeff_list, bvp_list, valid_list, b_pos):
    range_list[b_pos + 1] = (range_list[b_pos][0], range_list[b_pos + 1][1])
    del range_list[b_pos]

    while not len(bvp_list[b_pos][0]) == 0:
        bvp_list[b_pos + 1][0].insert(0, bvp_list[b_pos][0][-1])
        bvp_list[b_pos + 1][1].insert(0, bvp_list[b_pos][1][-1])

        del bvp_list[b_pos][0][-1]
        del bvp_list[b_pos][1][-1]

    del bvp_list[b_pos]
    del valid_list[b_pos]
    del coeff_list[b_pos]

    reg_data1 = copy.deepcopy(bvp_list[b_pos])
    normalize(reg_data1)
    coeff_list[b_pos] = get_coeffs_from_data(reg_data1, regularization, degree)


def increase_upper_boundary(range_list, coeff_list, bvp_list, b_pos, amount):
    assert (range_list[b_pos + 1][1] - range_list[b_pos + 1][0] - amount) > 0, \
        'Increasing boundary would eliminate next one!'

    range_list[b_pos] = (range_list[b_pos][0], range_list[b_pos][1] + amount)
    range_list[b_pos + 1] = (range_list[b_pos + 1][0] + amount, range_list[b_pos + 1][1])

    bvp_list[b_pos][0].extend(bvp_list[b_pos + 1][0][0:amount])
    bvp_list[b_pos][1].extend(bvp_list[b_pos + 1][1][0:amount])
    for i in range(amount):
        del bvp_list[b_pos + 1][0][0]
        del bvp_list[b_pos + 1][1][0]

    reg_data1 = copy.deepcopy(bvp_list[b_pos])
    normalize(reg_data1)
    coeff_list[b_pos] = get_coeffs_from_data(reg_data1, regularization, degree)

    reg_data2 = copy.deepcopy(bvp_list[b_pos + 1])
    normalize(reg_data2)
    coeff_list[b_pos + 1] = get_coeffs_from_data(reg_data2, regularization, degree)


def increase_lower_boundary(range_list, coeff_list, bvp_list, b_pos, amount):
    assert (range_list[b_pos][1] - range_list[b_pos][0] - amount) > 0, 'Increasing boundary would eliminate it!'

    range_list[b_pos] = (range_list[b_pos][0] + amount, range_list[b_pos][1])
    range_list[b_pos - 1] = (range_list[b_pos - 1][0], range_list[b_pos - 1][1] + amount)

    bvp_list[b_pos - 1][0].extend(bvp_list[b_pos][0][0:amount])
    bvp_list[b_pos - 1][1].extend(bvp_list[b_pos][1][0:amount])
    for i in range(amount):
        del bvp_list[b_pos][0][0]
        del bvp_list[b_pos][1][0]

    reg_data1 = copy.deepcopy(bvp_list[b_pos - 1])
    normalize(reg_data1)
    coeff_list[b_pos - 1] = get_coeffs_from_data(reg_data1, regularization, degree)

    reg_data2 = copy.deepcopy(bvp_list[b_pos])
    normalize(reg_data2)
    coeff_list[b_pos] = get_coeffs_from_data(reg_data2, regularization, degree)


def decrease_upper_boundary(range_list, coeff_list, bvp_list, b_pos, amount):
    assert (range_list[b_pos][1] - range_list[b_pos][0] - amount) > 0, 'Decreasing boundary would eliminate it!'

    range_list[b_pos] = (range_list[b_pos][0], range_list[b_pos][1] - amount)
    range_list[b_pos + 1] = (range_list[b_pos + 1][0] - amount, range_list[b_pos + 1][1])

    for i in range(amount):
        ts = bvp_list[b_pos][0][-1]
        bvp = bvp_list[b_pos][1][-1]

        bvp_list[b_pos + 1][0].insert(0, ts)
        bvp_list[b_pos + 1][1].insert(0, bvp)

        del bvp_list[b_pos][0][-1]
        del bvp_list[b_pos][1][-1]

    reg_data1 = copy.deepcopy(bvp_list[b_pos])
    normalize(reg_data1)
    coeff_list[b_pos] = get_coeffs_from_data(reg_data1, regularization, degree)

    reg_data2 = copy.deepcopy(bvp_list[b_pos + 1])
    normalize(reg_data2)
    coeff_list[b_pos + 1] = get_coeffs_from_data(reg_data2, regularization, degree)


def decrease_lower_boundary(range_list, coeff_list, bvp_list, b_pos, amount):
    assert (range_list[b_pos - 1][1] - range_list[b_pos - 1][0] - amount) > 0, \
        'Decreasing boundary would eliminate previous one!'

    range_list[b_pos] = (range_list[b_pos][0] - amount, range_list[b_pos][1])
    range_list[b_pos - 1] = (range_list[b_pos - 1][0], range_list[b_pos - 1][1] - amount)

    for i in range(amount):
        ts = bvp_list[b_pos - 1][0][-1]
        bvp = bvp_list[b_pos - 1][1][-1]

        bvp_list[b_pos][0].insert(0, ts)
        bvp_list[b_pos][1].insert(0, bvp)

        del bvp_list[b_pos - 1][0][-1]
        del bvp_list[b_pos - 1][1][-1]

    reg_data1 = copy.deepcopy(bvp_list[b_pos - 1])
    normalize(reg_data1)
    coeff_list[b_pos - 1] = get_coeffs_from_data(reg_data1, regularization, degree)

    reg_data2 = copy.deepcopy(bvp_list[b_pos])
    normalize(reg_data2)
    coeff_list[b_pos] = get_coeffs_from_data(reg_data2, regularization, degree)


def get_errors_for_boundary(range_list, coeff_list, bvp_list, valid_list, b_pos, l_pos, r_pos):
    bvp_data_norm = copy.deepcopy(bvp_list[b_idx + 1])
    normalize(bvp_data_norm)

    reg_error = 0
    for ts, bvp in zip(bvp_data_norm[0], bvp_data_norm[1]):
        pred_bvp = polynomial(ts, coeff_list[b_pos])
        reg_error += (bvp - pred_bvp) ** 2

    reg_error /= len(bvp_data_norm[0])

    coeff_error = 0
    avg_coeff = 0
    coeff_count = 0

    t_coeff_data = transpose(coeff_list[l_pos:r_pos + 1])

    target_middle = range_list[b_pos][0] + (range_list[b_pos][1] - range_list[b_pos][0]) / 2
    if r_pos - b_pos > b_pos - l_pos:
        edge_middle = range_list[r_pos][0] + (range_list[r_pos][1] - range_list[r_pos][0]) / 2
    else:
        edge_middle = range_list[l_pos][0] + (range_list[l_pos][1] - range_list[l_pos][0]) / 2

    for c_idx, c_set in enumerate(t_coeff_data):
        coeff_count += len(c_set)

        for b, c in enumerate(c_set):
            boundary_middle = (range_list[b + l_pos][1] + range_list[b + l_pos][0]) / 2

            if valid_list[b + l_pos] == 1:
                mul = 1 * (1 - abs((target_middle - boundary_middle) / (target_middle - edge_middle)))
            elif valid_list[b + l_pos] == 0:
                mul = 0 * (1 - abs((target_middle - boundary_middle) / (target_middle - edge_middle)))
            else:
                mul = 0

            avg_coeff += (coeff_list[b_pos][c_idx] - c) ** 2
            coeff_error += mul * (coeff_list[b_pos][c_idx] - c) ** 2

            # print('for tm =', target_middle, 'bm =', boundary_middle, 'mul is', mul, 'bpos coeff', coeff_list[b_pos][c_idx], 'compared coeff', c, 'contribution', mul * (coeff_list[b_pos][c_idx] - c) ** 2)

    coeff_error /= coeff_count
    coeff_error /= (avg_coeff / coeff_count)

    size_error = 0
    avg_size = 0
    size_count = 0

    for idx, (r_low, r_high) in enumerate(range_list[l_pos:r_pos + 1]):
        boundary_middle = (r_high + r_low) / 2

        if valid_list[l_pos + idx] == 1:
            mul = 1 * (1 - abs((target_middle - boundary_middle) / (target_middle - edge_middle)))
        elif valid_list[l_pos + idx] == 0:
            mul = 0 * (1 - abs((target_middle - boundary_middle) / (target_middle - edge_middle)))
        else:
            mul = 0

        size_error += mul * ((r_high - r_low) - (range_list[b_pos][1] - range_list[b_pos][0])) ** 2
        avg_size += (r_high - r_low)
        size_count += 1

    size_error /= size_count
    size_error /= (avg_size / size_count)

    return reg_error, coeff_error, size_error


if __name__ == '__main__':
    data_dirs = list(os.listdir('../data/output_consolidated'))
    regularization = .0001
    degree = 12

    for data_fname in data_dirs[0:1]:
        print('~~~ starting', data_fname, '~~~')
        raw_data = []
        boundary_range = []

        print('\tfinding boundaries...')
        with open('../data/output_consolidated/' + data_fname, 'r') as f:
            header = f.readline()

            last_boundary = 0

            line = f.readline().split(',')
            while line != ['']:
                if line[8] == '1':
                    # detected idi event, either write relevant stuff to output file or cache everything until later
                    boundary_range.append((last_boundary, len(raw_data) - 1))
                    last_boundary = len(raw_data) - 1

                raw_data.append(line)
                line = f.readline().strip().split(',')

            boundary_range.append((last_boundary, len(raw_data) - 1))

        numerical(raw_data)
        raw_data = transpose(raw_data)

        print('\tdetermining boundary validity')
        boundary_valid = [b_upper - b_lower < 100 for b_lower, b_upper in boundary_range]
        boundary_valid = list(map(lambda x: 1 if x else -1, boundary_valid))

        print('\textracting boundary data')
        boundary_data = []
        for b_lower, b_upper in boundary_range:
            b_data = [raw_data[0][b_lower:b_upper], raw_data[4][b_lower:b_upper]]
            boundary_data.append(b_data)

        print('\tcalculating boundary coefficients')
        boundary_coeffs = []
        for b_data, (b_lower, b_upper) in zip(boundary_data, boundary_range):
            norm_b_data = copy.deepcopy(b_data)
            normalize(norm_b_data)
            boundary_coeffs.append(get_coeffs_from_data(norm_b_data, regularization, degree))

        while any(i == -1 for i in boundary_valid):
            # determine best b_idx, l_idx, r_idx
            b_idx = 0
            l_idx = 0
            r_idx = 0

            for test_b_idx in range(len(boundary_range)):
                if boundary_valid[test_b_idx] == -1:
                    test_l_idx = test_b_idx
                    test_r_idx = test_b_idx

                    while test_l_idx > 0:  # and test_b_idx - test_l_idx < 50:
                        if boundary_valid[test_l_idx - 1] != -1:
                            test_l_idx -= 1
                        else:
                            break

                    while test_r_idx < len(boundary_range) - 1:  # and test_r_idx - test_b_idx < 50:
                        if boundary_valid[test_r_idx + 1] != -1:
                            test_r_idx += 1
                        else:
                            break

                    if test_r_idx - test_l_idx > r_idx - l_idx:
                        b_idx = test_b_idx
                        l_idx = test_l_idx
                        r_idx = test_r_idx

            # start working on the both-correct ones
            print('\tworking on boundary', boundary_range[b_idx], 'at index', b_idx)
            print('\t\tsurrounding valid indeces are', l_idx, r_idx)

            if r_idx - b_idx > b_idx - l_idx:
                # split on the right
                print('\t\tsplitting right')
                split_boundary_high(boundary_range, boundary_coeffs, boundary_data, boundary_valid, b_idx)

                reg_errors = []
                coeff_errors = []
                size_errors = []
                count = 0

                # try decrementing boundary until edge
                num_iterations = min(boundary_range[b_idx][1] - boundary_range[b_idx][0] - 1, 200)
                for _ in range(num_iterations):
                    # calculate error for that boundary
                    reg_error, coeff_error, size_error = get_errors_for_boundary(boundary_range, boundary_coeffs,
                                                                                 boundary_data,
                                                                                 boundary_valid, b_idx + 1, l_idx,
                                                                                 r_idx)

                    reg_errors.append(reg_error)
                    coeff_errors.append(coeff_error)
                    size_errors.append(size_error)

                    # vis_boundary_set(range_data, coeff_data, bvp_data)
                    count += 1
                    decrease_lower_boundary(boundary_range, boundary_coeffs, boundary_data, b_idx + 1, 1)

                # and also delete the one just below
                split = False
                if boundary_range[b_idx][1] - boundary_range[b_idx][0] == 1:
                    delete_boundary_merge_up(boundary_range, boundary_coeffs, boundary_data, boundary_valid, b_idx)
                    reg_error, coeff_error, size_error = get_errors_for_boundary(boundary_range, boundary_coeffs,
                                                                                 boundary_data,
                                                                                 boundary_valid, b_idx, l_idx, r_idx)

                    reg_errors.append(reg_error)
                    coeff_errors.append(coeff_error)
                    size_errors.append(size_error)
                    split = True

                sums = [r + c + s for r, c, s in zip(reg_errors, coeff_errors, size_errors)]
                min_idx = [idx for idx in range(len(sums)) if sums[idx] == min(sums)][0]

                print('\t\treg error range is', min(reg_errors), 'to', max(reg_errors))
                print('\t\tcoeff error range is', min(coeff_errors), 'to', max(coeff_errors))
                print('\t\tsize error range is', min(size_errors), 'to', max(size_errors))

                if min_idx - 1 >= 0:
                    print('\t\tsum to left is', sums[min_idx - 1])
                    print('\t\t\treg error is', reg_errors[min_idx - 1])
                    print('\t\t\tcoeff error is', coeff_errors[min_idx - 1])
                    print('\t\t\tsize error is', size_errors[min_idx - 1])

                print('\t\tmin sum is', sums[min_idx], 'at index', min_idx)
                print('\t\t\treg error is', reg_errors[min_idx])
                print('\t\t\tcoeff error is', coeff_errors[min_idx])
                print('\t\t\tsize error is', size_errors[min_idx])

                if min_idx + 1 < len(sums):
                    print('\t\tsum to right is', sums[min_idx + 1])
                    print('\t\t\treg error is', reg_errors[min_idx + 1])
                    print('\t\t\tcoeff error is', coeff_errors[min_idx + 1])
                    print('\t\t\tsize error is', size_errors[min_idx + 1])

                if min_idx > len(sums) - 10:
                    print('\t\tleaving lower boundary erased')
                    vis_boundary_set(boundary_range[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_coeffs[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_data[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_valid[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     reg_errors, coeff_errors, size_errors, min_idx)

                    # boundary_valid[b_idx] = 1
                else:
                    print('\t\trestoring lower boundary and shifting it up by', count - min_idx)
                    if split:
                        split_boundary_low(boundary_range, boundary_coeffs, boundary_data, boundary_valid, b_idx)
                        boundary_valid[b_idx] = -1

                    increase_lower_boundary(boundary_range, boundary_coeffs, boundary_data, b_idx + 1, count - min_idx)
                    vis_boundary_set(boundary_range[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_coeffs[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_data[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_valid[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     reg_errors, coeff_errors, size_errors, min_idx)

                    # boundary_valid[b_idx + 1] = 1

            else:
                # split on the left
                print('\t\tsplitting left')
                split_boundary_low(boundary_range, boundary_coeffs, boundary_data, boundary_valid, b_idx)

                reg_errors = []
                coeff_errors = []
                size_errors = []
                count = 0

                # try incrementing boundary until edge
                num_iterations = min(boundary_range[b_idx + 1][1] - boundary_range[b_idx + 1][0] - 1, 200)
                for _ in range(num_iterations):
                    # calculate error for that boundary
                    reg_error, coeff_error, size_error = get_errors_for_boundary(boundary_range, boundary_coeffs,
                                                                                 boundary_data,
                                                                                 boundary_valid, b_idx, l_idx, r_idx)

                    reg_errors.append(reg_error)
                    coeff_errors.append(coeff_error)
                    size_errors.append(size_error)

                    count += 1
                    increase_upper_boundary(boundary_range, boundary_coeffs, boundary_data, b_idx, 1)

                # and also try getting rid of the one just above
                split = False
                if boundary_range[b_idx + 1][1] - boundary_range[b_idx + 1][0] == 1:
                    delete_boundary_merge_down(boundary_range, boundary_coeffs, boundary_data, boundary_valid,
                                               b_idx + 1)
                    reg_error, coeff_error, size_error = get_errors_for_boundary(boundary_range, boundary_coeffs,
                                                                                 boundary_data,
                                                                                 boundary_valid, b_idx, l_idx, r_idx)

                    reg_errors.append(reg_error)
                    coeff_errors.append(coeff_error)
                    size_errors.append(size_error)
                    split = True

                sums = [r + c + s for r, c, s in zip(reg_errors, coeff_errors, size_errors)]
                min_idx = [idx for idx in range(len(sums)) if sums[idx] == min(sums)][0]

                print('\t\treg error range is', min(reg_errors), 'to', max(reg_errors))
                print('\t\tcoeff error range is', min(coeff_errors), 'to', max(coeff_errors))
                print('\t\tsize error range is', min(size_errors), 'to', max(size_errors))

                if min_idx - 1 >= 0:
                    print('\t\tsum to left is', sums[min_idx - 1])
                    print('\t\t\treg error is', reg_errors[min_idx - 1])
                    print('\t\t\tcoeff error is', coeff_errors[min_idx - 1])
                    print('\t\t\tsize error is', size_errors[min_idx - 1])

                print('\t\tmin sum is', min(sums), 'at index', min_idx)
                print('\t\t\treg error is', reg_errors[min_idx])
                print('\t\t\tcoeff error is', coeff_errors[min_idx])
                print('\t\t\tsize error is', size_errors[min_idx])

                if min_idx + 1 < len(sums):
                    print('\t\tsum to right is', sums[min_idx + 1])
                    print('\t\t\treg error is', reg_errors[min_idx + 1])
                    print('\t\t\tcoeff error is', coeff_errors[min_idx + 1])
                    print('\t\t\tsize error is', size_errors[min_idx + 1])

                # move to goal state
                if min_idx > len(sums) - 10:
                    print('\t\tleaving upper boundary erased')
                    vis_boundary_set(boundary_range[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_coeffs[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_data[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_valid[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     reg_errors, coeff_errors, size_errors, min_idx)

                    # boundary_valid[b_idx] = 1

                else:
                    print('\t\trestoring upper boundary and shifting it down by', count - min_idx)
                    if split:
                        split_boundary_high(boundary_range, boundary_coeffs, boundary_data, boundary_valid, b_idx)
                        boundary_valid[b_idx + 1] = -1

                    decrease_upper_boundary(boundary_range, boundary_coeffs, boundary_data, b_idx, count - min_idx)
                    vis_boundary_set(boundary_range[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_coeffs[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_data[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     boundary_valid[max(l_idx, b_idx - 10):min(r_idx + 1, b_idx + 11)],
                                     reg_errors, coeff_errors, size_errors, min_idx)

                    # boundary_valid[b_idx] = 1
