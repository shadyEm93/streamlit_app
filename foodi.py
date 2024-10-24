


import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px




# Load data
test = pd.read_csv(r"G:\sw\Anaconda3 (64-bit)\prolit\test.csv")
train = pd.read_csv(r"G:\sw\Anaconda3 (64-bit)\prolit\train.csv")

# Drop unnecessary columns
col = [train.columns[0], train.columns[1]]
train = train.drop(columns=col)

# Concatenate data
concat = pd.concat([train, test], ignore_index=True, axis=0)

# Create dictionary and DataFrame
data = {
    'res_lat': concat['Restaurant_latitude'],
    'res_lon': concat['Restaurant_longitude'],
    'des_lat': concat['Delivery_location_latitude'],
    'des_lon': concat['Delivery_location_longitude'],
    'Order_Date': concat['Order_Date']
}

df = pd.DataFrame(data)


# Calculate mean latitude and longitude
mean_lat = df['res_lat'].mean()
mean_lon = df['res_lon'].mean()

# Filter out outliers
threshold = 10 
df_filtered = df[(df['res_lat'] < mean_lat + threshold) & (df['res_lat'] > mean_lat - threshold) &
                 (df['res_lon'] < mean_lon + threshold) & (df['res_lon'] > mean_lon - threshold) &
                 (df['des_lat'] < mean_lat + threshold) & (df['des_lat'] > mean_lat - threshold) &
                 (df['des_lon'] < mean_lon + threshold) & (df['des_lon'] > mean_lon - threshold)]
df_filtered.loc[:, 'Order_Date']= concat.loc[df_filtered.index, 'Order_Date']


# Create a sample
df_subset = df_filtered.sample(n=2000, random_state=42)

# This select box can help the user see the data based on the date he chooses 
if st.markdown('##select orders date'):
   dates= df_subset['Order_Date'].unique()
   selected= st.selectbox('select the date:', dates)
   
   date= df_subset.loc[df_subset['Order_Date'] == selected]
   st.title('selected date')
   st.write(date)

# Create a scatter plot for the main map
fig = px.scatter_mapbox(date,
                        lat='res_lat',
                        lon='res_lon',
                        color_discrete_sequence=['blue'],
                        zoom=5,
                        height=600)

# Add delivery points to the map
fig.add_scattermapbox(lat=date['des_lat'],
                      lon=date['des_lon'],
                      mode='markers',
                      marker=dict(size=9, color='fuchsia'),
                      name='Delivery')

# Add lines to connect restaurants to delivery points
for i in range(len(date)):
    fig.add_scattermapbox(lat=[date['res_lat'].iloc[i], date['des_lat'].iloc[i]],
                          lon=[date['res_lon'].iloc[i], date['des_lon'].iloc[i]],
                          mode='lines',
                          line=dict(width=2, color='green'),
                          name=f'Route {i+1}')

# Update layout
fig.update_layout(mapbox_style="open-street-map",
                  mapbox_center={"lat": date['res_lat'].mean(),
                                 "lon": date['res_lon'].mean()},
                  mapbox_zoom= 5,
                  margin= {"r":0,"t":0,"l":0,"b":0})                            
                  

#Create another sample set for several charts
sample = concat.sample(n=80, random_state=42)

#Running the main map on stramlit webapp
st.header('Food Delivery Map')
st.markdown('Restaurants and Delivery Locations')
st.plotly_chart(fig)
st.markdown('In order to see the addresses well, please zoom in.')

# Create a sidebar 
st.sidebar.header('Options')

# Raw data gives a summery about the dataset
if st.sidebar.toggle('summery of the main data'):
    st.title('Raw Data')
    st.write(concat.head(20))    

#scatter plot for time taken and weather conditions
df = sample.dropna(subset=['Time_taken(min)', 'Weatherconditions'])

df= df.sort_values(by= 'Time_taken(min)')

scatter_fig = px.scatter(df, x='Weatherconditions', y='Time_taken(min)', 
                         title='Time Taken & Weather Conditions')

#box plot for type of order and delivery rating to show the distribution of delivery ratings for each type of order, highlighting medians and outliers.
box_df= sample

box_fig = px.box(box_df, x='Type_of_order', y='Delivery_person_Ratings', 
                 title='Delivery Rating Distribution by Type of Order',
                 labels={'Delivery_rating': 'Delivery Rating'},
                 color='Type_of_order')


# Heatmap for Correlation between time orderd and time taken
sample['Time_Orderd'] = pd.to_datetime(sample['Time_Orderd'], format='%H:%M:%S', errors='coerce')
if sample['Time_Orderd'].dtype == 'datetime64[ns]':
    sample['Time_Orderd_hour'] = sample['Time_Orderd'].dt.hour

# Correlation calcution needs numbers not strings 
sample['Time_taken(min)'].ffill(inplace=True)
sample['Time_taken(min)']= sample['Time_taken(min)'].str.extract(r'(\d+)')
sample.loc[:, 'Time_taken(min)'] = sample['Time_taken(min)'].astype(float).astype(int)

#Calcute the correlation  
correlation_matrix = sample[['Time_Orderd', 'Time_taken(min)']].corr()

# Create the plot
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap between Time Ordered (Hour) and Time Taken (min)')



# Create annother sample set for the last chaart
samplee = concat.sample(n=500, random_state=42)

# Scatter plot for multiple deliveries and delivery person ratings
# Data type should be float
samplee['multiple_deliveries'] = pd.to_numeric(samplee['multiple_deliveries'], errors='coerce')
samplee['Delivery_person_Ratings'] = pd.to_numeric(samplee['Delivery_person_Ratings'], errors='coerce')

samplee[['multiple_deliveries', 'Delivery_person_Ratings']] = samplee[['multiple_deliveries', 'Delivery_person_Ratings']].ffill()
samplee['multiple_deliveries'] = samplee['multiple_deliveries'].astype(int)
samplee = samplee[samplee['multiple_deliveries'].apply(lambda x: x.is_integer() if not pd.isna(x) else False)]

# Scatter plot for multiple deliveries and delivery person ratings
fig_scatter = px.scatter(samplee, x='multiple_deliveries', y='Delivery_person_Ratings', 
                         size='Delivery_person_Ratings', color='multiple_deliveries',
                         title='Multiple Deliveries vs. Delivery Person Ratings',
                         labels={'multiple_deliveries': 'Multiple Deliveries', 
                                 'Delivery_person_Ratings': 'Delivery Person Ratings'},
                         color_continuous_scale=px.colors.qualitative.Dark24,
                         hover_name='Delivery_person_ID')

fig_scatter.update_traces(marker=dict(size=12, opacity=0.8),
                          selector=dict(mode='markers'))




# Load the charts on streamlit 
if st.sidebar.toggle('TimeTaken/TimeOrderd'):
   st.title('Charts and Correlation')
   st.pyplot(plt)
   st.markdown('#### as you can see there is a weak relationship between the time taken and the time orderd based on this chart')
   
   
if st.sidebar.toggle('MultipleDeliveries/Rate'):
    st.title('Charts and Correlation')
    st.plotly_chart(fig_scatter)
    
    

if st.sidebar.toggle('Rate/OrderType'):
    st.title('chart and correlations')
    st.plotly_chart(box_fig)
    


if st.sidebar.toggle('TimeTaken/WeatherCondition'):
    st.title('chart and correlations')
    st.plotly_chart(scatter_fig)
    


























