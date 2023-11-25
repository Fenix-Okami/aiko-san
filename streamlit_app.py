import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome to Ray's Streamlit App! 👋")

st.sidebar.success("Select an app above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    """)
st.image("aiko_cover.png")
st.markdown(  """ 
    ## Meet AIKO-san
    She is designed to support your journey in learning Japanese. Let's enjoy mastering the language with AIKO-san!
    ### Features
    - **Japanese Fundamentals**: Learn grammar, vocabulary, and kanji for JLPT N5 and N4 levels.
    - **Interactive Conversations**: Engage in realistic dialogues to practice your Japanese.
    - **Cultural Insights**: Gain a deeper understanding of Japanese culture and customs.

    ### Getting Started
    - **Choose 'Chat with Aiko'** in the sidebar to begin interacting in Japanese at your chosen JLPT N-level.
    - Enhance your learning experience with **text-to-speech** and **text-to-image** features for dynamic illustrations and audio support.

    #### Want to Learn More?
    - For more details about this app, check out [streamlit.io](https://streamlit.io).
    - If you have any questions, you can find the commuity in [community forums](https://discuss.streamlit.io).
"""
)