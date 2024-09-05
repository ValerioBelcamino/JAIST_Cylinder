import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

base_path = '/home/s2412003/Shared/JAIST_Cylinder'
base_path = 'z:\\Shared\\JAIST_Cylinder'

available_models = ['IMU', 'videos', 'both']
which_model = 'IMU'

print(f'Using {which_model} model\n\n')

ground_truth = os.path.join(base_path, 'Segmented_Dataset_Realtime')

path_to_results = os.path.join(base_path, 'realtime_results', which_model)



action_names = ['linger', 'massaging', 'patting', 
                'pinching', 'press', 'pull', 
                'push', 'rub', 'scratching', 
                'shaking', 'squeeze', 'stroke', 
                'tapping', 'trembling', 'None']

action_names2 = ['pinching', 'tapping', 'stroke',
                'patting', 'shaking', 'trembling', 
                'pull', 'push', 'press', 'linger',
                'massaging', 'squeeze', 'scratching', 'rub'
                ]

action_dict = {action: i for i, action in enumerate(action_names)}
action_dict_inv = {i: action for i, action in enumerate(action_names)}


def compute_percentage_overlap(classified_labs, ground_truth_labs, use_none=False):
    equal_with_nones = 0
    equal_no_nones = 0
    nones = 0
    for cl, gt in zip(classified_labs, ground_truth_labs):
        if gt == 14:
            nones += 1

        if cl == gt:
            equal_with_nones += 1
            if gt != 14:
                equal_no_nones += 1

    return (equal_with_nones / len(classified_labs)) * 100, (equal_no_nones / (len(classified_labs) - nones)) * 100


# Function to blend color with white to create pastel colors
def pastel_color(rgba, factor=0.7):
    # Convert RGBA to RGB by ignoring the alpha channel
    rgb = np.array(rgba[:3]) 
    return tuple(rgb * factor + np.array([1, 1, 1]) * (1 - factor))

def plot_action_sequence(actions_start_end_list, length, saving_path):
    global action_names, action_names2, action_dict, action_dict_inv

    indices_in_seconds = [float(i)/30.0 for i in range(length)]


    # Generate 14 pastel rainbow colors
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, 14))
    pastel_colors = [pastel_color(color) for color in rainbow_colors]

    # print([colors(i) for i in range(14)])
    # exit()
    # Assign colors to each action
    action_colors = {f'{action_names2[i]}': pastel_colors[i] for i in range(14)}

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot each action as a box on the same line
    for i, action_start_end in enumerate(actions_start_end_list):
        for action, start, end in action_start_end:
            ax.barh(i, (end - start)/30.0, left=start/30.0, height=0.5, color=action_colors[action_names[action]], edgecolor='black', label=action_dict_inv[action], linewidth=0.5)

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_xticks(np.arange(0, max(indices_in_seconds), 60))
    ax.set_yticks([])  # Remove the y-ticks since all actions are on the same line
    ax.set_title('Actions Over Time')

    # Remove duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {label: handle for label, handle in zip(labels, handles)}
    unique_labels = {i: unique_labels[i] for i in action_names2}

    ax.legend(unique_labels.values(), unique_labels.keys(), title='Actions', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-10, max(indices_in_seconds)+10)


    # Display the plot
    plt.tight_layout()
    plt.savefig(saving_path)
    # plt.show()

def get_action_start_and_end(label_list):
    action_start_end = []
    current_action = None
    for i in range(len(label_list)):
        action_at_idx = label_list[i]
        # print(action_at_idx)
        if current_action is None:
            # print('start')
            current_action = action_at_idx
            start = i
        else:
            if current_action != action_at_idx:
                # print('end')
                if current_action != 14:
                    action_start_end.append((current_action, start, i-1))
                current_action = action_at_idx
                start = i
            if i == len(label_list)-1:
                # print('endend')
                if current_action != 14:
                    action_start_end.append((current_action, start, i))
    return action_start_end 




ground_truth_files = []
# for trials in os.listdir(ground_truth):
#     print(f'Processing {trials}')

for subfolder in os.listdir(os.path.join(ground_truth)):
    if subfolder == 'Labels':
        print(f'\tProcessing {os.path.join(ground_truth, subfolder)}')
        ground_truth_files.extend([os.path.join(ground_truth, subfolder, x) for x in os.listdir(os.path.join(ground_truth, subfolder)) if x.endswith('.csv') and 'annotations' in x])


print(f'\n\n{ground_truth_files}')
print(f'{len(ground_truth_files)}\n\n')

        
classified_labels_list, ground_truth_labels_list = [], []
for i, classified_file in enumerate(os.listdir(path_to_results)):
    # if i <= 3:
    #     continue
    if classified_file.endswith('.txt'):
        print(f'Processing {classified_file}')
        result_path = os.path.join(path_to_results, classified_file)
        with open(result_path, 'r') as f:
            classified_labels = f.readlines()
            classified_labels = [int(x) for x in classified_labels]
            classified_labels = np.array(classified_labels)
            # print(f'{result.shape}')
            # print('\n')

        # Get the corresponding ground truth
        trial = classified_file.split('_')[0]
        sequence_len = classified_file.split('_')[1]
        # print(f'{sequence_len}')
        # print(f'{ground_truth_files}')
        groun_truth_corresponding = [x for x in ground_truth_files if '_'.join([trial, sequence_len]) in x][0]
        print(f'{groun_truth_corresponding}\n')

        with open(groun_truth_corresponding, 'r') as f:
            ground_truth_labels = f.readlines()
            ground_truth_labels = [int(x.split(',')[1]) for x in ground_truth_labels]
            ground_truth_labels = np.array(ground_truth_labels)
            # print(f'{result}')
            # print(f'{result.shape}')

        # Get action start and end
        classified_start_end = get_action_start_and_end(classified_labels)
        # print(f'{classified_start_end}')
        # print(f'{len(classified_start_end)}')  

        ground_truth_start_end = get_action_start_and_end(ground_truth_labels)
        # print(f'{ground_truth_start_end}')
        # print(f'{len(ground_truth_start_end)}')  

        # Plot the action sequences
        plot_action_sequence([classified_start_end, ground_truth_start_end], int(sequence_len), os.path.join(path_to_results, f'{trial}_{sequence_len}_comparison.png'))

        classified_labels_list.extend(classified_labels)
        ground_truth_labels_list.extend(ground_truth_labels)

print(f'Total classified frames: {len(classified_labels_list)} ({len(classified_labels_list)/30} seconds)')

# Compute the percentage overlap
overlap, overlap_no_none = compute_percentage_overlap(classified_labels_list, ground_truth_labels_list, use_none=True)
print(f'Overlap with None: {overlap:.2f}')
print(f'Overlap without None: {overlap_no_none:.2f}\n\n')