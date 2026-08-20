import earthaccess

def authenticate_and_download_smap():
    """
    Authenticates with NASA Earthdata using earthaccess and downloads a sample of SMAP data.
    Requires Earthdata Login credentials (can be set via environment variables 
    EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or interactive prompt).
    """
    print("Authenticating with NASA Earthdata...")
    try:
        auth = earthaccess.login(strategy="interactive")
        if not auth:
            print("Failed to authenticate.")
            return
            
        print("Successfully authenticated. Searching for SMAP L4 Soil Moisture Data...")
        
        # Search for SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture
        # Shortname: SPL4SMGP
        results = earthaccess.search_data(
            short_name="SPL4SMGP",
            temporal=("2023-01-01", "2023-01-02"),
            bounding_box=(-120, 30, -100, 45), # Example: Western US
            count=5
        )
        
        print(f"Found {len(results)} granules. Downloading...")
        
        # Download files to a local data directory
        earthaccess.download(results, "./data/smap_raw")
        print("Download complete.")
        
    except Exception as e:
        print(f"Earthdata access error: {e}")
        print("Please ensure your NASA Earthdata login credentials are set up.")

if __name__ == '__main__':
    # authenticate_and_download_smap()
    print("This script is ready to download SMAP data via earthaccess.")
    print("Uncomment the execution block and ensure credentials are set before running.")
