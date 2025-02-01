import streamlit as st

prompt = st.chat_input('Ask anything...')

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        result = prompt
        st.write(result)
