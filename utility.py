#firstly we will import required libraries
import re
import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nlp = spacy.load("en_core_web_sm")
#then we will prepare a text extractor from pdf file
def pdf2txt(pdfFile):
    reader = PyPDF2.PdfReader(pdfFile)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + ""
    return text
#now we will list out all the skills we need to match with the resume
skills =["C",
    "C++",
    "Python",
    "Java",
    "JavaScript",
    "Go",
    "MATLAB",
    "Bash",
    "Dart",
    "TypeScript",
    "PHP",
    "Swift",
    "R",
    "Rust",

    "SQL",
    "MySQL",
    "MongoDB",
    "Oracle",
    "SQLite",
    "Firebase",
    "PostgreSQL",
    "Cassandra",
    "DynamoDB",

    "HTML5",
    "React.js",
    "Node.js",
    "CSS3",
    "Vue.js",
    "GraphQL",
    "jQuery",
    "Django",
    "Angular",
    "Express.js",
    "Next.js",
    "Flask",
    "Spring Boot",
    "Bootstrap",
    "WebSockets",
    "REST APIs",

    "AWS",
    "Azure",
    "Terraform",
    "OpenShift",
    "Serverless Architecture",
    "Google Cloud Platform (GCP)",
    "EC2",
    "S3",
    "RDS",
    "Kubernetes",
    "Cloud Functions",
    "Lambda",

    "Machine Learning",
    "Deep Learning",
    "Time Series Analysis",
    "NLP",
    "Natural Language Processing",
    "Supervised Learning",
    "Unsupervised Learning",
    "Model Deployment",
    "Recommendation Systems",
    "Feature Engineering",

    "Android Development",
    "iOS Development",
    "React Native",
    "SwiftUI",
    "Flutter",
    "Firebase Integration",
    "Jetpack Compose",

    "Object Oriented Programming",
    "Design Patterns",
    "Data Structures & Algorithms",
    "Unit Testing",
    "TDD (Test Driven Development)",
    "Microservices Architecture",

    "Git",
    "GitHub Actions",
    "Docker",
    "CI/CD Pipelines",
    "Kubernetes",
    "ELK Stack",
    "Grafana",
    "Ansible",
    "Prometheus",

    "ETL Pipelines",
    "Snowflake",
    "Hive",
    "Data Warehousing",
    "BigQuery",
    "Kafka",
    "Airflow",
    "Apache Hadoop",
    "Apache Spark",

    "OpenCV",
    "Scikit Learn",
    "PyTorch",
    "TensorFlow",
    "LightGBM",
    "Statsmodels",
    "XGBoost",
    "NLTK",
    "Keras",

    "Arduino",
    "Raspberry Pi",
    "IoT Systems",
    "Microcontrollers",
    "RTOS",
    "Embedded C",
    "VHDL",
    "Verilog",
    "PCB Design",

    "Excel (Advanced)",
    "Unity",
    "Figma",
    "Power BI",
    "JIRA",
    "Selenium",
    "Blender",
    "Apache JMeter",
    "Swagger"
]

#here we are using Tf IDF for checking text similarity
def similar(resume,job):
    vectorizer = TfidfVectorizer()
    vectors= vectorizer.fit_transform([resume,job])
    return cosine_similarity(vectors[0],vectors[1])[0][0]
#it will remove all the punctuation marks,symbols,space,and convert the uppercase to lowercase
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]"," ",text)
    return re.sub(r"\s+"," ",text).strip()
#it will extract the matching skills with the required skill set
def extractSkills(text):
    text = text.lower()
    return [s for s in skills if s in text]
#it will extract the number of years of experience
def extractExp(text):
    match = re.findall(r"(\d{1,2})\s*(year|years|yrs|yr)",text.lower())
    years= [int(m[0]) for m in match]
    return max(years) if years else 0
#we will normalize experience within range of 0 to 1
def scoreExp(candidate,required):
    if required==0:
        return 1
    return min(candidate/required,1)
#now the evaluation of resume  will take place
def evaluate(resume,jobDescription,reqExp):
    resClean=clean(resume)
    jobClean=clean(jobDescription)
    #for matching the skills if found
    jSkills=extractSkills(jobClean)
    rSkills=extractSkills(resClean)
    skillsScore=len(set(rSkills) & set(jSkills))/max(1,len(jSkills))
    #for checking experience
    candExp=extractExp(resClean)
    scoresExp=scoreExp(candExp,reqExp)
    #for checking text similarity
    scoreTxt=similar(resClean,jobClean)
    #weightage distribution
    score=(skillsScore*0.5)+(scoresExp*0.3)+(scoreTxt*0.2)
    #suggestions for remaining skills for improvement scope
    misSkills=list(set(jSkills) - set(rSkills))

    return{
        "skillsScore" : round(skillsScore,3),
        "expCand" : candExp,
        "scoresExp" : round(scoresExp,3),
        "scoreTxt" : round(scoreTxt,3),
        "score" : round(score,3),
        "misSkills" : misSkills,
    }
