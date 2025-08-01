import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr

class ResumeGenerator:
    def _init_(self):
        # Initialize all required components first
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Initialize LLM with your token
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            huggingfacehub_api_token="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",
            model_kwargs={
                "temperature": 0.3,
                "max_length": 2000
            }
        )
        
        self.vector_db = None
        self.resume_template = ""
        self.cover_letter_template = ""
        
        try:
            self.load_sample_documents()
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            raise

    def load_sample_documents(self):
        """Load and extract text from sample PDFs with error handling"""
        try:
            # Load resume PDF
            if not os.path.exists("Kritika_Resume (4).pdf"):
                raise FileNotFoundError("Resume PDF not found")
            
            with open("Kritika_Resume (4).pdf", "rb") as file:
                reader = PyPDF2.PdfReader(file)
                self.resume_template = "\n".join([page.extract_text() for page in reader.pages])
                if not self.resume_template.strip():
                    raise ValueError("Resume PDF is empty or couldn't be read")
            
            # Load cover letter PDF
            if not os.path.exists("COVER LETTER.pdf"):
                raise FileNotFoundError("Cover letter PDF not found")
                
            with open("COVER LETTER.pdf", "rb") as file:
                reader = PyPDF2.PdfReader(file)
                self.cover_letter_template = "\n".join([page.extract_text() for page in reader.pages])
                if not self.cover_letter_template.strip():
                    raise ValueError("Cover letter PDF is empty or couldn't be read")
            
            # Create vector database
            combined_text = f"RESUME TEMPLATE:\n{self.resume_template}\n\nCOVER LETTER TEMPLATE:\n{self.cover_letter_template}"
            chunks = self.text_splitter.split_text(combined_text)
            self.vector_db = FAISS.from_texts(chunks, self.embeddings)
            
        except Exception as e:
            print(f"Error in document processing: {str(e)}")
            raise

    def generate_resume(self, user_info):
        """Generate resume with error handling"""
        try:
            if not self.vector_db:
                return "Error: Document database not initialized"
                
            prompt_template = """Using this resume template format, create a personalized resume:
            
            TEMPLATE:
            {template}
            
            USER INFORMATION:
            {user_info}
            
            Instructions:
            1. Maintain exact template structure
            2. Fill all sections with user's information
            3. Keep professional tone
            4. Ensure consistent formatting"""
            
            prompt = PromptTemplate.from_template(prompt_template).format(
                template=self.resume_template,
                user_info=user_info
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
                chain_type="stuff"
            )
            
            result = qa_chain.run(prompt)
            return result if result else "Failed to generate resume (empty response)"
            
        except Exception as e:
            return f"Resume generation error: {str(e)}"

    def generate_cover_letter(self, user_info, job_description):
        """Generate cover letter with error handling"""
        try:
            if not self.vector_db:
                return "Error: Document database not initialized"
                
            prompt_template = """Create a cover letter using this template:
            
            TEMPLATE:
            {template}
            
            APPLICANT INFO:
            {user_info}
            
            JOB DESCRIPTION:
            {job_description}
            
            Instructions:
            1. Use exact template format
            2. Address key job requirements
            3. Highlight relevant qualifications"""
            
            prompt = PromptTemplate.from_template(prompt_template).format(
                template=self.cover_letter_template,
                user_info=user_info,
                job_description=job_description
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
                chain_type="stuff"
            )
            
            result = qa_chain.run(prompt)
            return result if result else "Failed to generate cover letter (empty response)"
            
        except Exception as e:
            return f"Cover letter generation error: {str(e)}"

def generate_documents(personal_info, job_description, generate_resume, generate_cover_letter):
    try:
        generator = ResumeGenerator()
        results = []
        
        if generate_resume:
            resume = generator.generate_resume(personal_info)
            results.append(f"=== GENERATED RESUME ===\n{resume}")
        
        if generate_cover_letter and job_description.strip():
            cover_letter = generator.generate_cover_letter(personal_info, job_description)
            results.append(f"\n=== GENERATED COVER LETTER ===\n{cover_letter}")
        
        return "\n\n".join(results) if results else "Please select documents to generate"
        
    except Exception as e:
        return f"Application error: {str(e)}"

iface = gr.Interface(
    fn=generate_documents,
    inputs=[
        gr.Textbox(label="Your Personal Information", lines=5,
                 placeholder="Full name, contact info, education, experience, skills..."),
        gr.Textbox(label="Job Description (for cover letter)", lines=5,
                 placeholder="Paste the complete job description..."),
        gr.Checkbox(label="Generate Resume", value=True),
        gr.Checkbox(label="Generate Cover Letter", value=True)
    ],
    outputs=gr.Textbox(label="Results", lines=20),
    title="Professional Resume Generator",
    description="Note: Ensure PDF templates are in the same folder"
)

if _name_ == "_main_":
    iface.launch()
