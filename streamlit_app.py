import streamlit as st

st.set_page_config(
    page_title="AI-ko",
    page_icon="🇯🇵",
)

st.write("# AI-ko")

st.sidebar.success("Select an app above.")

left_co, right_co = st.columns(2)
with left_co:
    st.image('aiko_cover.png', caption='愛子さん', width=300)
with right_co:
    st.markdown( """ 
        ## Meet 愛子さん
        She is designed to support your journey in learning Japanese!
        ### Available interactions
        - **☕ Aiko-chat**: Have a simple chat with her. she will try to be mindful of your current proficiency level. There are settings to add voice and illustrations
        - **📖 Aiko-test**: Aiko will try her best to challenge you with questions on the fly, and provide detailed explanations. She can be a bit clumsy but she's doing her best!
"""
)