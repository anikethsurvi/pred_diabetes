#core
import streamlit as st 

#EDA package
import pandas as pd

#VIZ package 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px 

#load data
def load_data(data):
	df = pd.read_csv(data)
	return df 

def run_eda_app():
	st.subheader("EXPLORATORY DATA ANALYSIS")
	df = load_data("C:/Users/hp/ml_streamlit/diabetes_data_upload.csv")
	df_encoded = load_data("data/diabetes_data_upload_clean.csv")
	freq_df = load_data("data/freqdist_of_age_data.csv")

	submenu = st.sidebar.selectbox("submenu",['Descriptive',"Plots"])
	if submenu == 'Descriptive':
		st.dataframe(df)

		with st.expander("Data Types Summary"):
			st.dataframe(df.dtypes)

		with st.expander("Descriptive summary"):
			st.dataframe(df_encoded.describe())

		with st.expander("Class Distribution"):
			st.dataframe(df['class'].value_counts())

		with st.expander("Gender Distribution"):
			st.dataframe(df['Gender'].value_counts())

	elif submenu == 'Plots':
		st.subheader("Plots")

		# layouts
		col1,col2 = st.columns([2,1])

		with col1:
			#gender distribution
			with st.expander("Distribution plots of Gender"):
				
				gen_df = df["Gender"].value_counts()
				gen_df = gen_df.reset_index()
				gen_df.columns = ["Gender Type", "Counts"]
				#st.dataframe(gen_df)

				p1 = px.pie(gen_df,names='Gender Type',values = 'Counts')
				st.plotly_chart(p1,use_container_width = True)

			# for class distribution
			with st.expander("Distribution plots of class"):
				fig = plt.figure()
				sns.countplot(df["class"])
				st.pyplot(fig)


			# frequency distribution
			with st.expander("Frequency distribution of age"):
				p2 = px.bar(freq_df,x='Age',y="count")
				st.plotly_chart(p2,use_container_width = True)



		with col2:
			with st.expander("Gender Distribution"):
				st.dataframe(gen_df)

			with st.expander("Class Distribution"):
				st.dataframe(df["class"].value_counts())

			# frequency distribution
			with st.expander("Frequency distribution of age"):
				st.dataframe(freq_df)

		with st.expander("Outlier Detection Plot"):

			p3 = px.box(df,x='Age',color = 'Gender')
			st.plotly_chart(p3)

		# corelation
		with st.expander("Correlation Plot"):
			corr_matrix = df_encoded.corr()
			fig = plt.figure(figsize=(20,10))
			sns.heatmap(corr_matrix,annot = True)
			st.pyplot(fig)

		st.subheader("Auto Ml Visualization")

		from PIL import Image
		with st.expander("Different Models Comparison using AutoML"):
			img = Image.open("C:/Users/hp/ml_streamlit/model_comare.png")
			st.image(img,use_column_width = True)


		with st.expander("Learning curve for Random Forest classifier"):
			img = Image.open("C:/Users/hp/ml_streamlit/learning_curve.png")
			st.image(img,use_column_width = True)

		with st.expander("ROC curve for Random Forest classifier"):
			img = Image.open("C:/Users/hp/ml_streamlit/roc_curve.png")
			st.image(img,use_column_width = True)

		with st.expander("confusion matrix"):
			img = Image.open("C:/Users/hp/ml_streamlit/cm.png")
			st.image(img,use_column_width = True)





