def perform_downsampling(file_path, encoding='latin1'):
    import pandas as pd
    import numpy as np
    
    # Output temporaneo, puoi decidere se salvarlo realmente o restituirlo direttamente
    output_path = "temp_downsampled.csv"

    timer_cols = ['Minutes', 'Seconds', 'Frames']
    label_cols = ['label', 'Activity', 'Result']

    df = pd.read_csv(file_path)

    if 'Minutes' not in df.columns:
        df = df.iloc[:, :6]
        df.columns = timer_cols + label_cols

    for col in timer_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=timer_cols).reset_index(drop=True)

    df['time'] = df['Minutes'] * 60 + df['Seconds'] + df['Frames'] / 30.3

    df = df.sort_values('time').reset_index(drop=True)

    t_start = df['time'].iloc[0]
    t_end = df['time'].iloc[-1]
    n_samples = int(np.floor((t_end - t_start) * 20)) + 1
    new_time = np.linspace(t_start, t_start + (n_samples - 1) * (1/20), n_samples)

    df_new = pd.DataFrame({'time': new_time})

    df_downsampled = pd.merge_asof(df_new, df, on='time', direction='nearest')

    df_downsampled['Minutes'] = (df_downsampled['time'] // 60).astype(int)
    df_downsampled['Seconds'] = (df_downsampled['time'] % 60).astype(int)
    df_downsampled['Frames'] = np.floor((df_downsampled['time'] % 1) * 20).astype(int)

    cols_out = ['Minutes', 'Seconds', 'Frames'] + label_cols + ['time']
    df_downsampled = df_downsampled[cols_out]

    # Puoi scegliere di restituire direttamente il DataFrame invece di salvarlo
    # df_downsampled.to_csv(output_path, index=False)
    
    return df_downsampled
