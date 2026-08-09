import pandas as pd

# Import all the necessary functions from your custom module
from asteroid_tools import (
    get_valid_input, 
    get_reducted_magnitude, 
    interpolating, 
    clustering, 
    plot_clustering, 
    periodogram, 
    plot_periodogram, 
    get_best_order, 
    plot_chi2, 
    extract_c0, 
    plot_fourier_fit, 
    find_parameter, 
    plot_phase_curve
)

if __name__ == "__main__":
    print("===== Starting the Program =====")
    print("Hint: Press 'Ctrl + C' at any time to force stop the program.\n")

    path_data = str(input("Enter the path for lightcurve data: "))
    path_ephem = str(input("Enter the path for ephemeris data: "))

    print("\nReading the data...")

    # Take the both data from path using pandas
    df = pd.read_csv(path_data)
    df.columns = ['jd', 'mag', 'err']
    df = df.dropna().sort_values(by='jd').reset_index(drop=True)
    df = df[df['err'] > 0]

    df_ephem = pd.read_csv(path_ephem)
    df_ephem.columns = ['jd', 'phase_angle', 'helio_dist', 'geo_dist']
    df_ephem = df_ephem.dropna().sort_values(by='jd').reset_index(drop=True)

    print("Both data has been readed succesfully!")

    # Meminta input dengan validasi dan perlindungan Ctrl+C
    is_reducted = get_valid_input("Do you want to remove the distance effect from magnitude (y/n)? ", str, ['y', 'n'])

    if is_reducted == 'y':
        df = get_reducted_magnitude(df, df_ephem)
        print("The distance effect has been removed!")

    # Step 1
    df_interpolated = interpolating(df, df_ephem)
    print("\nInterpolating phase angle... Success!")

    # Step 2
    is_clustering = get_valid_input("\nDo you want to manually input number of cluster (y/n)? ", str, ['y', 'n'])

    if is_clustering == 'y':
        num_cluster = get_valid_input("Input number of cluster: ", int)

        print("\nClustering...")
        df_clustered, clustering_model = clustering(df_interpolated, is_clustering, num_cluster)
    elif is_clustering == 'n':
        print("\nClustering...")
        df_clustered, clustering_model = clustering(df_interpolated, is_clustering)

    plot_clustering(df_clustered)
    
    # Step 3
    is_period = get_valid_input("\nDo you want to manually input the rotation period (y/n)? ", str, ['y', 'n'])

    if is_period == 'y':
        P = get_valid_input("Input asteroid's rotation period (in hours): ", float)
        P = P / 24
    elif is_period == 'n':
        print("\nFinding best rotation period...")
        plot_period, P, frequency, power = periodogram(df_clustered)
        plot_periodogram(frequency, power, plot_period)

    # Step 4
    is_order = get_valid_input("\nDo you want to manually input the Fourier order (y/n)? ", str, ['y', 'n'])

    if is_order == 'y':
        best_order = get_valid_input("Input fourier order: ", int)
    elif is_order == 'n':
        print("\nFinding best Fourier order...")
        best_order, chi2_list, orders = get_best_order(df_clustered, P)
        plot_chi2(chi2_list, orders)

    # Step 5
    print("\nFitting fourier series to lightcurve...")
    df_final, plot_data = extract_c0(df_clustered, P, best_order)
    print("Result: ")
    print(df_final.to_string())
    plot_fourier_fit(plot_data, P, best_order)

    # Step 6
    print("\nCalculating parameters...")
    params = find_parameter(df_final)
    plot_phase_curve(df_final, params)

    print("\n===== End of the Program =====")
