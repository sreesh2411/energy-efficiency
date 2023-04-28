import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import matplotlib.pyplot as plt  # charts
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import plotly.figure_factory as ff
import seaborn as sns # charts
import pickle


page_big_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #e5e5f7;
    opacity: 0.8;
    filter: bluer(20px);
    background: linear-gradient(#e66465, #9198e5);
}
</style>
"""



st.set_page_config(
    page_title="Energy Efficiency Project Dashboard",
    page_icon="🔋",
    layout="wide",
)

#st.markdown(page_big_img, unsafe_allow_html = True)

dataset_url = "https://raw.githubusercontent.com/sreesh2411/energy-efficiency/main/ENB2012_data.csv"
# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    df = pd.read_csv(dataset_url)
    old = ['X1','X2','X3','X4','X5','X6','X7','X8','Y1','Y2']
    new = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load']
    for i in range(len(old)):
        df.rename(columns = {old[i]: new[i]}, inplace = True)
    return df


df = get_data()

df1 = pd.read_csv('model_results.csv')
df2 = pd.read_csv('model_tuned_results.csv')

new = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution', 'Heating_Load', 'Cooling_Load']


menu = ['Dashboard', 'Data', 'Prediction', 'About']

choice = st.sidebar.selectbox("Menu", menu)



if choice == 'Data':
    st.subheader("Detailed Data View")
    st.dataframe(df)

    st.subheader('Data Description')
    st.text("• Relative Compactness: This is the relative compactness of the building, which is defined as the ratio of the building's volume to the volume of an equivalent cuboid that encloses the building. This variable ranges from 0.62 to 0.98.\n• Surface Area - m²: This is the total surface area of the building, including walls, roof, and windows. This variable ranges from 514.5 to 808.5 square meters.\n• Wall Area - m²: This is the total area of the building's walls. This variable ranges from 245 to 416.5 square meters.\n• Roof Area - m²: This is the total area of the building's roof. This variable ranges from 110.25 to 220.5 square meters.\n• Overall Height - m: This is the height of the building. This variable ranges from 3.5 to 7.0 meters.\n• Orientation - 2:North, 3:East, 4:South, 5:West: This is the orientation of the building. The values 2, 3, 4, and 5 represent North, East, South, and West orientations, respectively.\n Glazing Area - 0%, 10%, 25%, 40% (of floor area): This is the total glazing area of the building, expressed as a percentage of the floor area. This variable can take on one of four values: 0%, 10%, 25%, or 40%.\n• Glazing Area Distribution (Variance) - 1:Uniform, 2:North, 3:East, 4:South, 5:West: This is the distribution of glazing area across the building. The values 1, 2, 3, 4, and 5 represent a uniform distribution and North, East, South, and West distributions, respectively.\n• Heating Load - kWh/m²: This is the heating load of the building, expressed in kilowatt-hours per square meter. This variable ranges from 6.01 to 43.1 kWh/m².\n• Cooling Load - kWh/m²: This is the cooling load of the building, expressed in kilowatt-hours per square meter. This variable ranges from 10.9 to 48.03 kWh/m².")

elif choice == 'Prediction':
    #loading the saved models 
    ridge_heat = pickle.load(open('/Users/sreeshreddy/Desktop/stat656/Project/Task_4/ridge_model_heat.sav', 'rb'))
    ridge_cool = pickle.load(open('/Users/sreeshreddy/Desktop/stat656/Project/Task_4/ridge_model_cool.sav', 'rb'))
    scaler = pickle.load(open('/Users/sreeshreddy/Desktop/stat656/Project/Task_4/scaler.sav', 'rb'))

    def load_predict(Relative_Compactness, Surface_Area, Wall_Area, Roof_Area, Overall_Height, house_orientation, Glazing_Area, house_glazing_Area_Dist):
    
    # processing user input
        Orientation = 2 if house_orientation =='North' else 3 if house_orientation == 'East' else 4 if house_orientation =='South' else 5
        
        Glazing_Area_Dist = 1 if house_glazing_Area_Dist == 'Uniform' else 2 if house_glazing_Area_Dist =='North' else 3 if house_glazing_Area_Dist =='East' else 4 if house_glazing_Area_Dist =='South' else 5 if house_glazing_Area_Dist =='West' else 0
        
        lists = [Relative_Compactness, Surface_Area, Wall_Area, Roof_Area, Overall_Height, Orientation, Glazing_Area, Glazing_Area_Dist]
        
        df = pd.DataFrame(lists).transpose()
        # scaling the data
        scaled_df = scaler.transform(df)
        # making predictions using the trained model
        prediction_heat_load = ridge_heat.predict(scaled_df)
        prediction_cool_load = ridge_cool.predict(scaled_df)
        result1 = float(prediction_heat_load)
        result2 = float(prediction_cool_load)
        return result1, result2


    style = """<div style='background-color:pink; padding:12px'>
              <h1 style='color:black'> Building Energy Prediction</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)
    left, right = st.columns((2,2))
    Relative_Compactness = left.number_input('Enter the relative compactness value',
                                  step =0.1, format="%.2f", value=0.62)
    Surface_Area = right.number_input('Enter the surface area of the building',
                                  step=1.0, format='%.2f', value= 500.84)
    Wall_Area = left.number_input('Enter the wall area of the building',
                                           step=1.0, format='%.2f', value=295.0)
    Roof_Area = right.number_input('Enter the Roof Area of the building',
                                     step=1.0, format='%.2f', value=110.25)
    Overall_Height = left.number_input('Enter the overall height of the building (in meters)',
                                       step=1.0, format='%.1f', value=7.0)
    Orientation = st.selectbox('Orientation of the building?',
                    ('North', 'East', 'South', 'West'))
    Glazing_Area = left.number_input('Enter the glazing area of the building',  step=0.1,
                                   format='%.2f',value=0.25)
    Glazing_Area_Dist =st.selectbox('Glazing Area Distribution of the building?',
                    ('Uniform', 'North', 'East', 'South', 'West', 'Unknown'))   
    button = st.button('Predict')
    
    if button:
        
        # make prediction
        result = load_predict(Relative_Compactness, Surface_Area, Wall_Area, Roof_Area, Overall_Height, Orientation, Glazing_Area, Glazing_Area_Dist)
        st.success(f'Heating Load of the building: {result[0]}')
        st.success(f'Cooling Load of the building: {result[1]}')


    
elif choice == 'About':
    st.subheader("About: Stat 656 Project: Heating & Coolings Buildings")
    
    st.info("Built with Streamlit and Python for the Course Project of STAT 656: Applied Analytics")
    st.text("Gunjan Joshi")
    st.text("Udbhav Srivastava")
    st.text("K. Sreesh Reddy")

   
elif choice == 'Dashboard':

    # dashboard title
    st.title("Stat 656 Project: Heating & Coolings Buildings")


    col1, col2 = st.columns(2)
    ##Distribution of all variables
    # Set up the plot

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    ## Distribution of variables

    with col1:
        fig = make_subplots(rows=4, cols=2, vertical_spacing = 0.15)

        fig.add_trace(
            go.Histogram(x=df['Relative_Compactness'], name = "Relative Compactness"),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=df['Surface_Area'], name = "Surface Area"),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=df['Wall_Area'], name = "Wall Area"),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=df['Roof_Area'], name = "Roof Area"),
            row=2, col=2
        )
        fig.add_trace(
            go.Histogram(x=df['Overall_Height'], name = "Overall Height"),
            row=3, col=1
        )
        fig.add_trace(
            go.Histogram(x=df['Orientation'], name = "Orientation"),
            row=3, col=2
        )
        fig.add_trace(
            go.Histogram(x=df['Glazing_Area'], name = "Glazing Area"),
            row=4, col=1
        )
        fig.add_trace(
            go.Histogram(x=df['Glazing_Area_Distribution'], name = "Glazing Area Distribution"),
            row=4, col=2
        )


                          
        fig.update_layout(title_text='Distribution of All Variables')
        st.plotly_chart(fig) 





        # Set up the plot
        fig = px.histogram(df, x=["Heating_Load", "Cooling_Load"], nbins=30, barmode="overlay", opacity=0.5)

        # Set the plot title and axis labels
        fig.update_layout(title="Distribution of Heating Load and Cooling Load", xaxis_title="Load (kWh/m2)", yaxis_title="Frequency")

        # Show the plot
        st.plotly_chart(fig)

        hist_data = [df['Cooling_Load'], df['Heating_Load']]
        group_labels = ['Cooling Load', 'Heating Load']
        fig = ff.create_distplot(hist_data, group_labels)
        fig.update_layout(title="Distribution of Heating Load and Cooling Load", xaxis_title="Load (kWh/m2)", yaxis_title="Frequency")

        st.write(fig)

        ridge_heat = pickle.load(open('/Users/sreeshreddy/Desktop/stat656/Project/Task_4/ridge_model_heat.sav', 'rb'))
        ridge_cool = pickle.load(open('/Users/sreeshreddy/Desktop/stat656/Project/Task_4/ridge_model_cool.sav', 'rb'))
        scaler = pickle.load(open('/Users/sreeshreddy/Desktop/stat656/Project/Task_4/scaler.sav', 'rb'))
        
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split

        
        df3 = get_data()
        df4 = get_data()
        # Generate some random data
        
        X = df3.drop(['Heating_Load', 'Cooling_Load'], axis=1)
        y = df3['Heating_Load']
        # Split the data into training and testing sets
        X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fit the ridge regression model
        ridge1 = Ridge(alpha=0.01)
        ridge1.fit(X_train_h, y_train_h)

        # Predict the output values for the testing set
        y_pred_h = ridge1.predict(X_test_h)

        # Plot the fitted line
        scatter_fig1 = px.scatter(x = y_test_h,y = y_pred_h, title = 'Fitted Line for Heating Load')
        # fig2 = px.scatter([min(y_test), max(y_test)], [min(y_test), max(y_test)])

        df5 = pd.DataFrame({'True Values': [min(y_test_h), max(y_test_h)], 'Predictions': [min(y_pred_h), max(y_pred_h)]})

        line_fig1 = px.line(df5, x='True Values', y='Predictions', color_discrete_sequence=['red', 'blue'])

        scatter_fig1.add_traces(line_fig1.data)

        


        scatter_fig1.update_layout(xaxis_title='True Values', yaxis_title='Predictions')


        st.plotly_chart(scatter_fig1)
        
        
        X = df4.drop(['Heating_Load', 'Cooling_Load'], axis=1)
        y = df4['Cooling_Load']
        # Split the data into training and testing sets
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fit the ridge regression model
        ridge2 = Ridge(alpha=0.01)
        ridge2.fit(X_train_c, y_train_c)

        # Predict the output values for the testing set
        y_pred_c = ridge2.predict(X_test_c)

        # Plot the fitted line
        scatter_fig = px.scatter(x = y_test_c,y = y_pred_c, title = 'Fitted Line for Cooling Load')
        # fig2 = px.scatter([min(y_test), max(y_test)], [min(y_test), max(y_test)])

        df6 = pd.DataFrame({'True Values': [min(y_test_c), max(y_test_c)], 'Predictions': [min(y_test_c), max(y_test_c)]})

        line_fig = px.line(df6, x='True Values', y='Predictions', color_discrete_sequence=['red', 'blue'])


        

        scatter_fig.add_traces(line_fig.data)



        scatter_fig.update_layout(xaxis_title='True Values', yaxis_title='Predictions')

        # Show the plot
        st.plotly_chart(scatter_fig)
        
        # st.plotly_chart(fig2)





    with col2:
    # Correlation Matrix
        fig, ax = plt.subplots(figsize = (10,10))
        ax.set_title('Correlation Matrix')
        sns.heatmap(df.corr(), ax=ax, fmt='.2f', annot = True)

        st.write(fig)

        option = st.selectbox(
            'Select the model you want to view results for:',
            ('XGBoost', 'Ridge', 'RF', 'CART', 'KNN', 'Lasso', 'ElasticNet'))

        #st.write('You selected:', option)

        
        st.dataframe(df1.loc[df1['Model_Names'] == option])
        st.dataframe(df2.loc[df2['Model_Names'] == option])

        st.set_option('deprecation.showPyplotGlobalUse', False)

        fig =  plt.figure(figsize=(15, 12))
        sns.barplot(x='RMSE_Test', y=df2['Model_Names'], data=df2, color="lightgreen")
        plt.xlabel('RMSE Values')
        plt.ylabel('Model Names')
        plt.title('RMSE_Test for All Models')
        st.pyplot()

        df5 = get_data()
        df6 = get_data()


        X_heat = pd.DataFrame(df5.drop(['Heating_Load', 'Cooling_Load'], axis=1))
        y_heat = pd.DataFrame(df5['Heating_Load'])

        features = X_heat.columns
        target = y_heat.columns

        coef1 = pd.DataFrame({'feature': features, 'importance': np.abs(ridge1.coef_)})

        # Sort the data frame by feature importance
        coef1 = coef1.sort_values('importance', ascending=False)

        # Create a bar plot of the feature importances
        fig1 = px.bar(coef1, x='feature', y='importance', color = 'importance', color_discrete_map={'Relative_Compactness':'green',
                                 'Glazing_Area':'cyan',
                                 'Overall_Height':'yellow',})

        # Set the title and axis labels
        fig1.update_layout(title='Ridge Regression Feature Importances (Heating Load)', xaxis_title='Feature', yaxis_title='Importance')

        # Show the plot
        fig1.update_yaxes(type='log')

        st.plotly_chart(fig1)

        X_cool = pd.DataFrame(df6.drop(['Heating_Load', 'Cooling_Load'], axis=1))
        y_cool = pd.DataFrame(df6['Heating_Load'])

        features = X_cool.columns
        target = y_cool.columns

        coef2 = pd.DataFrame({'feature': features, 'importance': np.abs(ridge2.coef_)})

        # Sort the data frame by feature importance
        coef2 = coef2.sort_values('importance', ascending=False)

        # Create a bar plot of the feature importances
        fig2 = px.bar(coef2, x='feature', y='importance', color = 'importance')

        # Set the title and axis labels
        fig2.update_layout(title='Ridge Regression Feature Importances (Cooling Load)', xaxis_title='Feature', yaxis_title='Importance')

        fig2.update_yaxes(type='log')

        # Show the plot
        st.plotly_chart(fig2)
                
                
        
    # ## Distribution of Heating and Cooling Load
    # # Set up the plot
    # 
    # sns.set_style("darkgrid")
    # fig, ax = plt.subplots()

    # # Plot the distribution of Heating_Load
    # sns.histplot(data=df, x="Heating_Load", kde=True, color="blue", alpha=0.5, ax=ax)

    # # Plot the distribution of Cooling_Load
    # sns.histplot(data=df, x="Cooling_Load", kde=True, color="red", alpha=0.5, ax=ax)

    # # Set the plot title and axis labels
    # ax.set_title("Distribution of Heating_Load and Cooling_Load")
    # ax.set_xlabel("Load (kWh/m2)")
    # ax.set_ylabel("Frequency")

    # # Show the plot
    # st.pyplot(fig)

    # from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    # from sklearn.exceptions import ConvergenceWarning
    # from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, ElasticNet
    # from sklearn.neighbors import KNeighborsRegressor
    # from sklearn.tree import DecisionTreeRegressor
    # from xgboost import XGBRegressor
    # from sklearn.preprocessing import LabelEncoder
    # from sklearn.metrics import mean_squared_error
    # from sklearn.linear_model import LinearRegression
    # from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate

  #   def grab_col_names(dataframe, cat_th=15, car_th=20):
  # #Catgeorical Variable Selection
  #     cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category","object","bool"]]
  #     num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["uint8","int64","float64"]]
  #     cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category","object"]]
  #     cat_cols = cat_cols + num_but_cat
  #     cat_cols = [col for col in cat_cols if col not in cat_but_car]

  #     #Numerical Variable Selection
  #     num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["uint8","int64","float64"]]
  #     num_cols = [col for col in num_cols if col not in cat_cols]

  #     return cat_cols, num_cols, cat_but_car, num_but_cat





  #   def outlier_thresholds(dataframe,col_name,q1=0.10,q3=0.90):
  #     quartile1 = dataframe[col_name].quantile(q1)
  #     quartile3 = dataframe[col_name].quantile(q3)
  #     interquartile_range = quartile3 - quartile1
  #     low_limit = quartile1 - 1.5 * interquartile_range
  #     up_limit = quartile3 + 1.5 * interquartile_range
  #     return low_limit,up_limit




  #   def check_outlier(dataframe, col_name):
  #     low_limit,up_limit = outlier_thresholds(dataframe,col_name)
  #     if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
  #       return True
  #     else:
  #       return False




  #   def replace_with_thresholds(dataframe, col_name):
  #     low_limit, up_limit = outlier_thresholds(dataframe, col_name)
  #     dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
  #     dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit





  #   def solve_outliers(dataframe, target):
  #     cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)
  #     for col in num_cols:
  #       if col!=target:
  #         #print(col, check_outlier(dataframe, col))
  #         if check_outlier(dataframe, col):
  #           replace_with_thresholds(dataframe, col)

  #   def create_base_model(dataframe, target, plot=False, save_results=False):
  #     names = []
  #     train_rmse_results = []
  #     test_rmse_results = []
  #     train_r2_scores = []
  #     test_r2_scores = []
  #     X = dataframe.drop(target, axis=1)
  #     y = dataframe[target]
  #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
  #     #print(X_train.isnull().values.any())
  #     #print(X_test.isnull().values.any())
  #     #print(y_train.isnull().values.any())
  #     #print(y_test.isnull().values.any())
  #     models = [('LR', LinearRegression()),
  #               ("Ridge", Ridge()),
  #               ("Lasso", Lasso()),
  #               ("ElasticNet", ElasticNet()),
  #               ('KNN', KNeighborsRegressor()),
  #               ('CART', DecisionTreeRegressor()),
  #               ('RF', RandomForestRegressor()),
  #               ("XGBoost", XGBRegressor(objective='reg:squarederror'))]

  #     #print("###################### Mean and Std(Target Variable) ######################")
  #     #print("Mean: " , dataframe[target].mean())
  #     #print("\n")
  #     #print("Std: ", dataframe[target].std())
  #     #print("###################### Model Results ######################")

  #     for name, regressor in models:
  #       regressor.fit(X_train, y_train)
  #       y_train_pred = regressor.predict(X_train)
  #       y_test_pred = regressor.predict(X_test)
  #       # RMSE
  #       train_rmse_result = np.sqrt(mean_squared_error(y_train, y_train_pred))
  #       test_rmse_result = np.sqrt(mean_squared_error(y_test, y_test_pred))
  #       train_rmse_results.append(train_rmse_result)
  #       test_rmse_results.append(test_rmse_result)
  #       # score
  #       train_r2_score = regressor.score(X_train, y_train)
  #       test_r2_score = regressor.score(X_test, y_test)
  #       train_r2_scores.append(train_r2_score)
  #       test_r2_scores.append(test_r2_score)
  #       # Model names
  #       names.append(name)
      
  #     model_results = pd.DataFrame({'Model_Names': names,
  #                                   'RMSE_Train': train_rmse_results,
  #                                   'RMSE_Test': test_rmse_results,
  #                                   'R2_score_Train': train_r2_scores,
  #                                   'R2_score_Test': test_r2_scores
  #                                   }).set_index("Model_Names")
  #     model_results=model_results.sort_values(by="RMSE_Test", ascending=True)
  #     #print(model_results)

  #     # if plot:
  #     #   plt.figure(figsize=(15, 12))
  #     #   sns.barplot(x='RMSE_Test', y=model_results.index, data=model_results, color="lightblue")
  #     #   plt.xlabel('RMSE Values')
  #     #   plt.ylabel('Model Names')
  #     #   plt.title('RMSE_Test for All Models')
  #     #   plt.show()
      
  #     if save_results:
  #       model_results.to_csv("model_results.csv")
      
  #     return model_results

  #   model_tuned_results = create_base_model(df, ["Heating_Load","Cooling_Load"], plot=True, save_results=True)

  #   def create_model_tuned(dataframe, target, plot=True, save_results=True):
  #     names = []
  #     train_rmse_results = []
  #     test_rmse_results = []
  #     train_r2_scores = []
  #     test_r2_scores = []
  #     best_params = []
  #     X = dataframe.drop(target, axis=1)
  #     y = dataframe[target]
  #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

  #     ridge_params = {"alpha": 10**np.linspace(10,-2,100)}

  #     lasso_params = {"alpha": 10**np.linspace(10,-2,100)}

  #     enet_params = {"alpha": 10**np.linspace(10,-2,100)*0.5}

  #     knn_params = {"n_neighbors": np.arange(1,50,1)}

  #     cart_params = {"min_samples_split": range(2,100),
  #                     "max_leaf_nodes": range(2,10)}

  #     xgboost_params = {"colsample_bytree": [0.4, 0.5 ,0.6],
  #                         "n_estimators": [200, 500, 1000],
  #                         "max_depth": [3,5,8],
  #                         "learning_rate": [0.1, 0.01]}

  #     rf_params = {"max_depth": [5, 8, 15, None],
  #                 "max_features": [5, 7, "auto"],
  #                 "min_samples_split": [8, 15, 20],
  #                 "n_estimators": [200, 500]}

  #     regressors = [("Ridge", Ridge(), ridge_params),
  #                 ("Lasso", Lasso(), lasso_params),
  #                 ("ENet", ElasticNet(), enet_params),
  #                 ("KNN", KNeighborsRegressor(), knn_params),
  #                 ("CART", DecisionTreeRegressor(), cart_params),
  #                 ("XGBoost", XGBRegressor(objective="reg:squarederror"), xgboost_params),
  #                 ("RF", RandomForestRegressor(), rf_params)]

  #     for name, regressor, params in regressors:
  #       #print(f"#################### {name} ####################")

  #       gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X_train, y_train)

  #       final_model = regressor.set_params(**gs_best.best_params_).fit(X_train, y_train)
  #       train_rmse_result = np.mean(np.sqrt(-cross_val_score(final_model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")))
  #       test_rmse_result = np.mean(np.sqrt(-cross_val_score(final_model, X_test, y_test, cv=10, scoring="neg_mean_squared_error")))
  #       train_rmse_results.append(train_rmse_result)
  #       test_rmse_results.append(test_rmse_result)
  #       train_r2_score = final_model.score(X_train, y_train)
  #       test_r2_score = final_model.score(X_test, y_test)
  #       train_r2_scores.append(train_r2_score)
  #       test_r2_scores.append(test_r2_score)
  #       #print(f"RMSE_Train: {round(train_rmse_result, 4)} , RMSE_Test: {round(test_rmse_result, 4)} , R2_Train: {round(train_r2_score, 3)} , R2_Test: {round(test_r2_score, 3)} (Tuned Model) ({name}) ")

  #       #print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
  #       best_params.append(gs_best.best_params_)
  #       names.append(name)
      
  #     model_tuned_results = pd.DataFrame({'Model_Names': names,
  #                                   'RMSE_Train': train_rmse_results,
  #                                   'RMSE_Test': test_rmse_results,
  #                                   'R2_score_Train': train_r2_scores,
  #                                   'R2_score_Test': test_r2_scores,
  #                                   "best_params": best_params
  #                                   }).set_index("Model_Names")
  #     model_tuned_results=model_tuned_results.sort_values(by="RMSE_Test", ascending=True)
  #     print(model_tuned_results)

  #     # if plot:
  #     #   plt.figure(figsize=(15, 12))
  #     #   sns.barplot(x='RMSE_Test', y=model_tuned_results.index, data=model_tuned_results, color="r")
  #     #   plt.xlabel('RMSE Values')
  #     #   plt.ylabel('Model Names')
  #     #   plt.title('RMSE_Test for All Models')
  #     #   st.show()
      
  #     if save_results:
  #       model_tuned_results.to_csv("model_tuned_results.csv")
      
  #     return model_tuned_results

  #   model_tuned_results = create_model_tuned(df, ["Heating_Load","Cooling_Load"], plot=True, save_results=True)

    