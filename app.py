#designing dashboard for UI
from utility import *
import os
import joblib
from sentence_transformers import SentenceTransformer
modelPath="model/jobRoleModel.pkl"
embed_model=SentenceTransformer("all-MiniLM-L6-v2")
jobRoleModel=joblib.load("model/jobRoleModel.pkl")
#checking if file exists
if not os.path.exists(modelPath):
    raise FileNotFoundError(f"File not found at {modelPath}!Run 'training.py' to create the model or place the file in the 'model' folder.")
else:
    jobRoleModel=joblib.load(modelPath)
    print("Model loaded succesfully!")
st.set_page_config(page_title="Smart Job Finder",layout="wide")
st.title("Job Match System using Resume")
st.write("Upload your resume(PDF) to compare it with any job description.")
tab1,tab2=st.tabs(["Match your Resume","Recruiter Dashboard"])
#for coloured headings
st.markdown("""
<style>
    .stMetric {
        background-color: #f5f7fa;
        padding: 10px;
        border-radius: 10px
    }
    h3 {
        color: #2c3e50;
    }
</style>
""",unsafe_allow_html=True)
#settings window
st.sidebar.header("Recruiter Filters")
minScore=st.sidebar.slider("Minimum Match Score",0.0,1.0,0.5)
sortOrder=st.sidebar.radio("Sort Order",["High to Low","Low to High"])
listSkills=st.sidebar.checkbox("List all the matched skills",value=True)

#tab 1 is designed for job seeker
with tab1:
    st.header("Job Seeker Panel")
    jDescription=st.text_area("Paste Job Description Here: ")
    #checking the validity of input
    if jDescription and jDescription.strip()!="":
        try:
            jEmbedding=jobRoleModel.encode([jDescription])
            st.success("Job description encoded successfully!")
        except Exception as e:
            st.error(f"Error encoding job description: {e}")
            jEmbedding=None
    else:
        st.warning("Please enter the job description.")
        jEmbedding=None
    if jEmbedding is not None:
        pass
    reqExp=st.number_input("Required Experience(in years):",0,30,0)
    pdfFile=st.file_uploader("Upload PDF of your Resume!",type=["pdf"])
    #button for submission
    if st.button("Analyze Resume"):
        if not pdfFile:
            st.error("Kindly upload the resume PDF.")
        #analyzing the resume uploaded
        else:
            resTxt=pdf2txt(pdfFile)
            result=evaluate(resTxt,jDescription,reqExp)
            col1,col2,col3,col4=st.columns(4)
            col1.metric("Skill Match",result["skillsScore"])
            col2.metric("Experience Score",result["scoresExp"])
            col3.metric("Text Similarity",result["scoreTxt"])
            col4.metric("Final Score",result["score"])
            st.write(f"**Extracted Experience(in years):** {result['expCand']}years")
            #in case of skills missing
            if result["misSkills"]:
                st.warning("Suggested skills to add for the  role: ")
                for skill in result["misSkills"]:
                    st.write(f"- {skill}")
            else:
                st.success("Your skills fully match the job recruitment.")
#tab 2 is designed for Industry mode(Recruiters)
with tab2:
    st.header("Recruiter Panel")
    jDescription=st.text_area("Paste Job Description for candidates: ")
    reqExp=st.number_input("Required Experience for job(in years):",0,30,0)
    resumes=st.file_uploader("Upload PDFs of resume of candidates",type=["pdf"],accept_multiple_files=True)
    #button for ranking
    if st.button("Rank the candidates!"):
        st.write("Rank button clicked")
        if jDescription.strip()=="":
            st.warning("Please enter Job Description to proceed!")
        elif resumes  is None or len(resumes)==0:
            st.warning("Upload at least resume of one candidate!")
        else:
            jEmbedding=embed_model.encode([jDescription])
            results=[]
            for r in resumes:
                txt=pdf2txt(r)
                evaluation=evaluate(txt,jDescription,reqExp)
                results.append({"Name": r.name,"Score": evaluation})
            #sorting the candidates based on their final scores
            results=[r for r in results if r.get("score",{}).get("final")is not None]
            results=sorted(results,key=lambda x: x.get["score"].get["final"],reverse=True)
            st.subheader("Ranked Candidates")
            for idx,res in enumerate(results,start=1):
                with st.container():
                    st.markdown(f"### Rank {idx}: {r['name']}")
                    col1,col2=st.columns(2)
                    with col1:
                        st.metric("Match Score",f"{r['score']}%")
                        st.progress(r['score']/100)
                    with col2:
                        if r['score']>=75:
                            st.success("Excellent Match")
                        elif r['score']>=50:
                            st.warning("Good Match")
                        else:
                            st.error("Low Match")
                    with st.expander("Detailed Analysis"):
                        st.write("Matched Skills:", r.get("skillsScore",[]))
                        st.write("Missing Skills:", r.get("misSkills",[]))
                    if listSkills:
                        st.write("Matched Skills:", r.get("skillsScore",[]))
                    if r["score"]>=minScore:
                        st.write(r)
                    st.divider()
#for footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:gray;">
        <small>
        &copy 2025 | Resume Job Role Classifier
        <br>
        Created by Mahak Singh
        </small>
    </div>
    """,
    unsafe_allow_html=True
    )

                