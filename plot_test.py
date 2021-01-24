import pickle
import random 
from plot import plot_logs_columns

if __name__ == '__main__':        

    with open('data/las_data_DTSM.pickle', 'rb') as f:
        las_dict = pickle.load(f)

    # plot some random las
    key = random.choice(list(las_dict.keys()))

    key_ = key.split('/')[1]
    plot_logs_columns(las_dict[key], 
                        well_name=key_,
                        plot_show=True,)
                        # plot_return=False,
                        # plot_save_file_name=key_,
                        # plot_save_path='plots',
                        # plot_save_format=['png', 'html'])

    # plot all las
    for key in las_dict.keys():
        key_ = key.split('/')[1]
        plot_logs_columns(las_dict[key], 
                        well_name=key,
                        plot_show=False,
                        plot_return=False,
                        plot_save_file_name=key_,
                        plot_save_path='plots',
                        plot_save_format=['png', 'html'])
