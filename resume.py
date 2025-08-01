import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr

class ResumeGenerator:
    def __init__(self):
        
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")  # Using Gemini Pro
        self.vector_db = None
        self.resume_template = ""
        self.cover_letter_template = ""
        
        
        self.load_sample_documents()
        
    def load_sample_documents(self):
        """Load and extract text from sample resume and cover letter PDFs"""
        
        with open("sample_resume.pdf", "rb") as file:
            reader = PyPDF2.PdfReader(file)
            self.resume_template = "\n".join([page.extract_text() for page in reader.pages])
        
        
        with open("sample_cover_letter.pdf", "rb") as file:
            reader = PyPDF2.PdfReader(file)
            self.cover_letter_template = "\n".join([page.extract_text() for page in reader.pages])
        
        
        combined_text = f"RESUME TEMPLATE:\n{self.resume_template}\n\nCOVER LETTER TEMPLATE:\n{self.cover_letter_template}"
        chunks = self.text_splitter.split_text(combined_text)
        self.vector_db = FAISS.from_texts(chunks, self.embeddings)
    
    def generate_resume(self, user_info):
        """Generate a resume based on user input and sample template"""
        prompt_template = PromptTemplate(
            input_variables=["user_info", "template"],
            template="""Using the following resume template format, create a personalized resume based on the user's information.
            
            Resume Template:
            {template}
            
            User Information:
            {user_info}
            
            Generate a professional resume maintaining:
            1. The exact same format and section structure as the template
            2. Professional tone and language
            3. All relevant sections filled with user's specific details
            4. Consistent formatting throughout
            """
        )
        
        prompt = prompt_template.format(user_info=user_info, template=self.resume_template)
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff"
        )
        return qa_chain.run(prompt)
    
    def generate_cover_letter(self, user_info, job_description):
        """Generate a matching cover letter based on user input and sample template"""
        prompt_template = PromptTemplate(
            input_variables=["user_info", "job_description", "template"],
            template="""Using the following cover letter template format, create a personalized cover letter based on the user's information and job description.
            
            Cover Letter Template:
            {template}
            
            User Information:
            {user_info}
            
            Job Description:
            {job_description}
            
            Generate a professional cover letter that:
            1. Maintains the exact same format as the template
            2. Directly addresses the job requirements
            3. Highlights 3-5 most relevant qualifications
            4. Uses professional business letter format
            5. Matches the style and tone of the resume
            """
        )
        
        prompt = prompt_template.format(
            user_info=user_info,
            job_description=job_description,
            template=self.cover_letter_template
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff"
        )
        return qa_chain.run(prompt)

def generate_documents(personal_info, job_description, generate_resume, generate_cover_letter):
    generator = ResumeGenerator()
    results = []
    
    if generate_resume:
        resume = generator.generate_resume(personal_info)
        results.append(f"=== GENERATED RESUME ===\n{resume}")
    
    if generate_cover_letter and job_description:
        cover_letter = generator.generate_cover_letter(personal_info, job_description)
        results.append(f"\n=== GENERATED COVER LETTER ===\n{cover_letter}")
    
    if not results:
        return "Please select at least one document to generate and provide job description for cover letter."
    
    return "\n\n".join(results)

iface = gr.Interface(
    fn=generate_documents,
    inputs=[
        gr.Textbox(label="Your Personal Information", 
                  placeholder="Include: Full name, contact info, education, work experience, skills, projects, etc."),
        gr.Textbox(label="Job Description (for cover letter)", 
                  placeholder="Paste the job description you're applying for"),
        gr.Checkbox(label="Generate Resume", value=True),
        gr.Checkbox(label="Generate Cover Letter", value=True)
    ],
    outputs=gr.Textbox(label="Generated Documents", lines=20),
    title="AI Resume & Cover Letter Generator (Powered by Gemini)",
    description="Upload your information and get professionally formatted documents based on our templates"
)

if __name__ == "__main__":
    iface.launch()