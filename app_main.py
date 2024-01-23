import streamlit as st 
import streamlit.components.v1 as stc

from eda_app import run_eda_app
from ml_app import run_ml_app


html_temp = '''
		<div style = "background-color : #3872fb;padding:10px;border-radius:10px">
		<h1 style = "color: white;text-align:center">Early Stage Diabetes Risk Data App</h1>
		<h2 style = "color : white;text-align:center;">Diabetes</h2>
		</div>
'''
# set PAGE CONFIGURATION
page_config = {"page_title":"Diabetes","page_icon":":smiley","layout":"centered"}
st.set_page_config(**page_config,initial_sidebar_state = 'expanded')

def main():
	stc.html(html_temp)

	menu = ['Home','Exploratory Data Analysis','Machine Learning']
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		st.write("""
			### Early Stage Diabetes Risk Predictor App
			This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
			#### Datasource
				- https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
			#### App Content
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section: ML Predictor App

			""")

	elif choice == "Exploratory Data Analysis":
		run_eda_app()

	else:
		run_ml_app()

	

if __name__ == '__main__':
	main()