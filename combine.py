import os


def process_data(datafolder, outputfile):
    with open(datafolder + '/ACC.csv', 'r') as f_acc, \
            open(datafolder + '/BVP.csv', 'r') as f_bvp, \
            open(datafolder + '/EDA.csv', 'r') as f_eda, \
            open(datafolder + '/HR.csv', 'r') as f_hr, \
            open(datafolder + '/IBI.csv', 'r') as f_ibi, \
            open(datafolder + '/TEMP.csv', 'r') as f_temp, \
            open(datafolder + '/tags.csv', 'r') as f_tags, \
            open(outputfile, 'w') as f_out:
        # first get the output file parameters from the individual data files
        acc_init_ts = float(f_acc.readline().split(',')[0])
        bvp_init_ts = float(f_bvp.readline().split(',')[0])
        eda_init_ts = float(f_eda.readline().split(',')[0])
        hr_init_ts = float(f_hr.readline().split(',')[0])
        ibi_init_ts = float(f_ibi.readline().split(',')[0])
        temp_init_ts = float(f_temp.readline().split(',')[0])

        acc_init_hz = float(f_acc.readline().split(',')[0])
        bvp_init_hz = float(f_bvp.readline().split(',')[0])
        eda_init_hz = float(f_eda.readline().split(',')[0])
        hr_init_hz = float(f_hr.readline().split(',')[0])
        temp_init_hz = float(f_temp.readline().split(',')[0])

        init_ts = min(acc_init_ts, bvp_init_ts, eda_init_ts, hr_init_ts, ibi_init_ts, temp_init_ts)
        init_hz = max(acc_init_hz, bvp_init_hz, eda_init_hz, hr_init_hz, temp_init_hz)

        # then, we start iterating through the data files and consolidating them
        f_out.write('timestamp,acc_x,acc_y,acc_z,bvp,eda,hr,temp,ibi,tag\n')

        acc_data = f_acc.readlines()
        bvp_data = f_bvp.readlines()
        eda_data = f_eda.readlines()
        hr_data = f_hr.readlines()
        ibi_data = f_ibi.readlines()
        temp_data = f_temp.readlines()
        tags_data = f_tags.readlines()

        acc_idx = 0
        bvp_idx = 0
        eda_idx = 0
        hr_idx = 0
        ibi_idx = 0
        temp_idx = 0
        tags_idx = 0

        current_ts = init_ts

        next_ibi_time = ibi_init_ts + float(ibi_data[ibi_idx].split(',')[0])

        if len(tags_data) > 0:
            next_tag_time = float(tags_data[tags_idx])
        else:
            next_tag_time = 0

        while acc_idx < len(acc_data) or \
                bvp_idx < len(bvp_data) or \
                eda_idx < len(eda_data) or \
                hr_idx < len(hr_data) or \
                ibi_idx < len(ibi_data) or \
                temp_idx < len(temp_data) or \
                tags_idx < len(tags_data):

            combined_data_row = str(current_ts) + ','

            if acc_idx < len(acc_data) and acc_init_ts + (acc_idx / acc_init_hz) == current_ts:
                combined_data_row += acc_data[acc_idx].strip()
                acc_idx += 1

                combined_data_row += ','
            else:
                combined_data_row += ',,,'

            if bvp_idx < len(bvp_data) and bvp_init_ts + (bvp_idx / bvp_init_hz) == current_ts:
                combined_data_row += bvp_data[bvp_idx].strip()
                bvp_idx += 1

            combined_data_row += ','

            if eda_idx < len(eda_data) and eda_init_ts + (eda_idx / eda_init_hz) == current_ts:
                combined_data_row += eda_data[eda_idx].strip()
                eda_idx += 1

            combined_data_row += ','

            if hr_idx < len(hr_data) and hr_init_ts + (hr_idx / hr_init_hz) == current_ts:
                combined_data_row += hr_data[hr_idx].strip()
                hr_idx += 1

            combined_data_row += ','

            if temp_idx < len(temp_data) and temp_init_ts + (temp_idx / temp_init_hz) == current_ts:
                combined_data_row += temp_data[temp_idx].strip()
                temp_idx += 1

            combined_data_row += ','

            if ibi_idx < len(ibi_data) and next_ibi_time == current_ts:
                combined_data_row += '1'
                ibi_idx += 1

                if ibi_idx < len(ibi_data):
                    next_ibi_time = ibi_init_ts + float(ibi_data[ibi_idx].split(',')[0])
                else:
                    next_ibi_time = 0

            combined_data_row += ','

            if tags_idx < len(tags_data) and next_tag_time == current_ts:
                combined_data_row += '1'
                tags_idx += 1

                if tags_idx < len(tags_data):
                    next_tag_time = float(tags_data[tags_idx])
                else:
                    next_tag_time = 0

            combined_data_row += '\n'
            f_out.write(combined_data_row)

            next_ts = current_ts + (1 / init_hz)

            # process ibi and tag events: first if they occur in the same interval
            if current_ts < next_ibi_time < next_ts and current_ts < next_tag_time < next_ts:
                if next_ibi_time < next_tag_time:
                    ibi_data_row = str(ibi_init_ts + float(ibi_data[ibi_idx].split(',')[0]))
                    ibi_data_row += ',,,,,,,,1,\n'
                    f_out.write(ibi_data_row)

                    ibi_idx += 1
                    if ibi_idx < len(ibi_data):
                        next_ibi_time = ibi_init_ts + float(ibi_data[ibi_idx].split(',')[0])
                    else:
                        next_ibi_time = 0

                    tags_data_row = tags_data[tags_idx].strip()
                    tags_data_row += ',,,,,,,,,1\n'
                    f_out.write(tags_data_row)

                    tags_idx += 1
                    if tags_idx < len(tags_data):
                        next_tag_time = float(tags_data[tags_idx])
                    else:
                        next_tag_time = 0
                else:
                    tags_data_row = tags_data[tags_idx].strip()
                    tags_data_row += ',,,,,,,,,1\n'
                    f_out.write(tags_data_row)

                    tags_idx += 1
                    if tags_idx < len(tags_data):
                        next_tag_time = float(tags_data[tags_idx])
                    else:
                        next_tag_time = 0

                    ibi_data_row = str(ibi_init_ts + float(ibi_data[ibi_idx].split(',')[0]))
                    ibi_data_row += ',,,,,,,,1,\n'
                    f_out.write(ibi_data_row)

                    ibi_idx += 1
                    if ibi_idx < len(ibi_data):
                        next_ibi_time = ibi_init_ts + float(ibi_data[ibi_idx].split(',')[0])
                    else:
                        next_ibi_time = 0
            else:
                if current_ts < next_ibi_time < next_ts:
                    ibi_data_row = str(ibi_init_ts + float(ibi_data[ibi_idx].split(',')[0]))
                    ibi_data_row += ',,,,,,,,1,\n'
                    f_out.write(ibi_data_row)

                    ibi_idx += 1
                    if ibi_idx < len(ibi_data):
                        next_ibi_time = ibi_init_ts + float(ibi_data[ibi_idx].split(',')[0])
                    else:
                        next_ibi_time = 0

                if current_ts < next_tag_time < next_ts:
                    tags_data_row = tags_data[tags_idx].strip()
                    tags_data_row += ',,,,,,,,,1\n'
                    f_out.write(tags_data_row)

                    tags_idx += 1
                    if tags_idx < len(tags_data):
                        next_tag_time = float(tags_data[tags_idx])
                    else:
                        next_tag_time = 0

            f_out.flush()
            current_ts = next_ts


for data in os.listdir('../data/WristBandData/'):
    print('consolidating folder', data)
    process_data('../data/WristBandData/' + data, '../data/output/' + data + '.csv')
    print('done with', data)
