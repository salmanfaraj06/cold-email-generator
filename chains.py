import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()
os.getenv("GROQ_API_KEY")

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links, recipient_name, company_name, recipient_position, your_name, your_position, your_company, your_email, your_phone):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are {your_name}, a {your_position} at {your_company}. {your_company} is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to {recipient_name}, {recipient_position} at {company_name}, regarding the job mentioned above describing the capability of {your_company} 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase {your_company}'s portfolio: {link_list}
            Do not provide a preamble.
            Include a polite closing with your contact details:
            Email: {your_email}
            Phone: {your_phone}
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": str(job),
            "link_list": links,
            "recipient_name": recipient_name,
            "company_name": company_name,
            "recipient_position": recipient_position,
            "your_name": your_name,
            "your_position": your_position,
            "your_company": your_company,
            "your_email": your_email,
            "your_phone": your_phone
        })
        return res.content

if __name__ == "__main__":
    print((os.getenv("GROQ_API_KEY")))
