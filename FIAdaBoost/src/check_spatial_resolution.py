import pandas as pd

df = pd.read_csv('data/raw/baseline_spatial_dataset_philippines_2024.csv')
print('Total rows:', len(df))
print('Unique locations:', df[['lat', 'lon']].drop_duplicates().shape[0])
print('Lat range:', df['lat'].min(), 'to', df['lat'].max())
print('Lon range:', df['lon'].min(), 'to', df['lon'].max())

print('\nDavao area subset (6.5-8.0 lat, 124.5-126.5 lon):')
davao = df[(df['lat'] >= 6.5) & (df['lat'] <= 8.0) & (df['lon'] >= 124.5) & (df['lon'] <= 126.5)]
print('Davao points:', len(davao))
print('Davao unique:', davao[['lat', 'lon']].drop_duplicates().shape[0])

if len(davao) > 0:
    print('\nSample Davao coordinates:')
    print(davao[['lat', 'lon', 'GHI_mean_2024']].head(15))
    
    # Check spacing between points
    lats = sorted(davao['lat'].unique())
    lons = sorted(davao['lon'].unique())
    if len(lats) > 1:
        print('\nLat spacing (degrees):', lats[1] - lats[0] if len(lats) > 1 else 'N/A')
    if len(lons) > 1:
        print('Lon spacing (degrees):', lons[1] - lons[0] if len(lons) > 1 else 'N/A')
