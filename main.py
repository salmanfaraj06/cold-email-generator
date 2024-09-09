import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from utils import clean_text
from chains import Chain
from portfolio import Portfolio


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Mail Generator")
    st.write('This is a simple tool to generate cold emails for your sales outreach.')
    url_input = st.text_input('Enter the URL of the recipient\'s website:', 'https://jobs.nike.com/job/R-38322')
    company_name = st.text_input('Enter the name of the recipient\'s company:', 'Nike')
    recipient_name = st.text_input('Enter the name of the recipient:', 'John Doe')
    recipient_email = st.text_input('Enter the email of the recipient:', 'salman.faraj06@gmail.com')
    recipient_position = st.text_input('Enter the position of the recipient:', 'CEO')
    your_name = st.text_input('Enter your name:', 'Salman Faraj')
    your_position = st.text_input('Enter your position:', 'Data Scientist')
    your_company = st.text_input('Enter your company:', 'Axle')
    your_email = st.text_input('Enter your email:', 'salman.20221380@iit.ac.lk')
    your_phone = st.text_input('Enter your phone number:', '0771234567')

    submit_button = st.button('Generate Email')

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links, recipient_name, company_name, recipient_position, your_name, your_position, your_company, your_email, your_phone)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)