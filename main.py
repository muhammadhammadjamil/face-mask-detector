import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from fastai.vision.all import *
import streamlit as st
import PIL
import torchvision.transforms as T


learn = load_learner('C:/Users/hammad jamil/Downloads/models1.pkl')

def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    html_title = """  
	<div style="text-align:center;"> 
		<h1>Mask or Not</h1>
	</div>
	"""
    st.markdown(html_title, unsafe_allow_html=True)
    st.subheader("Detect whether person is with_masks are Not")
    #learn = learner()
    uploaded_file = st.file_uploader("Choose a image file", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        st.image(uploaded_file, width=200)
        our_image =PILImage.create(uploaded_file)  # converting to an Image object
        pred = learn.predict(our_image)
        st.success(pred)
    footer = """
	<div style="position:fixed; text-align:center; bottom:0px; right:0px; left:0px; background-color:grey" markdown="1">
		<h4>Built with Streamlit using <a href= "https://www.fast.ai">fast.ai</a></h4>
		<p><a href="https://github.com/muhammadhammadjamil">Github</a> | &copyapsal</p>
	</div>
	"""
    st.markdown(footer, unsafe_allow_html=True)


if __name__ == '__main__':
    main()