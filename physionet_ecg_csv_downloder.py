import wfdb
import pandas as pd


def download_physionet_ecg_csv(record_name, dataset_name='mitdb', lead_name=None, output_filename=None):
    """
    Generalized function to download ECG data from PhysioNet and save as CSV.
    
    Parameters:
    -----------
    record_name : str
        The record identifier (e.g., '233', '212')
    dataset_name : str, default='mitdb'
        The PhysioNet dataset name (e.g., 'mitdb', 'incartdb', 'ltafdb')
    lead_name : str, optional
        Specific lead to extract (e.g., 'MLII', 'V5'). If None, saves all leads.
    output_filename : str, optional
        Custom output filename. If None, defaults to '{dataset_name}_{record_name}.csv'
    
    Returns:
    --------
    pd.DataFrame
        The downloaded ECG data as a DataFrame
    
    Example:
    --------
    >>> df = download_physionet_ecg_csv('233', dataset_name='mitdb', lead_name='MLII')
    >>> df = download_physionet_ecg_csv('100', dataset_name='incartdb')
    """
    
    try:
        # 1. Download and read the record from PhysioNet
        print(f"Downloading record {record_name} from {dataset_name}...")
        record = wfdb.rdrecord(record_name, pn_dir=dataset_name)
        
        # 2. Convert the signal to a Pandas DataFrame
        df = pd.DataFrame(record.p_signal, columns=record.sig_name)
        
        # 3. Add a time column (in seconds)
        df.insert(0, 'Time_s', [i / record.fs for i in range(len(df))])
        
        # 4. Extract specific lead if requested
        if lead_name:
            if lead_name not in df.columns:
                available_leads = [col for col in df.columns if col != 'Time_s']
                raise ValueError(f"Lead '{lead_name}' not found. Available leads: {available_leads}")
            df = df[['Time_s', lead_name]]
            print(f"Extracted lead: {lead_name}")
        
        # 5. Set default output filename if not provided
        if output_filename is None:
            lead_suffix = f"_{lead_name}" if lead_name else ""
            output_filename = f"{dataset_name}_{record_name}{lead_suffix}.csv"
        
        # 6. Save to CSV
        df.to_csv(output_filename, index=False)
        print(f"Record saved to {output_filename}")
        
        return df
    
    except Exception as e:
        print(f"Error downloading record: {e}")
        return None


# Example usage:
if __name__ == "__main__":
    # Download a single lead from mitdb
    df1 = download_physionet_ecg_csv('233', dataset_name='mitdb', lead_name='MLII')
    
    # Download all leads from a record
    df2 = download_physionet_ecg_csv('212', dataset_name='mitdb')
    
    # Download from a different dataset
    # df3 = download_physionet_ecg_csv('100', dataset_name='incartdb', lead_name='V5')
