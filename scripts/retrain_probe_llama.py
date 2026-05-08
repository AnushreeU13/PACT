"""
AU-Probe training — regression on redaction ratio (paired approach).

Dataset construction:
  For each of 127 base prompts:
    - Original prompt        → target = 0.0  (no tags, no redaction)
    - Pipeline output        → target = redacted_chars / total_chars

The probe learns to predict the fraction of text covered by [REDACTED ...]
tags directly from the Llama embedding. This encodes "more redaction →
higher score" as the training objective rather than a binary boundary.

At inference, score = clip(w · embedding + b, 0, 1)

Cache: pickle dict keyed by prompt text — caches both originals and
pipeline outputs.

Requires Ollama running with llama3.1:8b.
Run from repo root:  python scripts/retrain_probe_llama.py
"""

from __future__ import annotations

import os
import pickle
import re
import sys
import time

import numpy as np
import requests
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.pipeline_collect import sequential_redaction_pipeline

ROOT            = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PROBE    = os.path.join(ROOT, "data", "au_probe", "linearprobe_layer_32.pt")
CACHE_FILE      = os.path.join(ROOT, "data", "au_probe", "training_cache_v5.pkl")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("LOCAL_LLM_MODEL_NAME", "llama3.1:8b")

REDACTED_RE = re.compile(r"\[REDACTED[^\]]*\]", re.IGNORECASE)

MODULE_SETTINGS: dict[str, bool] = {
    "identity":    True,
    "location":    True,
    "demographic": True,
    "financial":   True,
    "health":      False,
}


def count_tags(text: str) -> int:
    return len(REDACTED_RE.findall(text))


def redaction_ratio(text: str) -> float:
    redacted_chars = sum(len(t) for t in REDACTED_RE.findall(text))
    return redacted_chars / max(len(text), 1)




# ---------------------------------------------------------------------------
# CLEAN prompts — no PII; pipeline leaves these unchanged → label 0
# ---------------------------------------------------------------------------
CLEAN_PROMPTS: list[str] = [
    "What is the best way to learn Python programming?",
    "Explain how photosynthesis works in simple terms.",
    "What are the best practices for writing clean code?",
    "How does compound interest work over 30 years?",
    "Can you recommend a good book on data privacy?",
    "How do I improve my public speaking skills?",
    "What causes inflation in an economy?",
    "Explain the concept of a relational database.",
    "What are the benefits of regular exercise?",
    "How does the human immune system fight viruses?",
    "How do solar panels convert sunlight to electricity?",
    "What is the significance of the Turing test?",
    "How does encryption protect data in transit?",
    "What are some effective study techniques for graduate students?",
    "How do neural networks learn from data?",
    "What is the difference between machine learning and deep learning?",
    "How should I prepare for a job interview at a tech company?",
    "What is the difference between term life and whole life insurance?",
    "How does the stock market work for a complete beginner?",
    "What is the difference between a REST API and a GraphQL API?",
    "How do I negotiate a salary increase at my current job?",
    "What are the main provisions of GDPR?",
    "What is the difference between a will and a living trust?",
    "How does the human digestive system work?",
    "What are the key principles of machine learning fairness?",
    "Explain the concept of zero-knowledge proofs.",
    "What is the difference between supervised and unsupervised learning?",
    "How does blockchain technology work?",
    "What are the most common cybersecurity threats today?",
    "Explain how DNS resolution works.",
    "What is the purpose of a credit score and how is it calculated?",
    "How do I set up a basic budget for the first time?",
    "What is the difference between a debit card and a credit card?",
    "How does the Federal Reserve control inflation?",
    "What are index funds and why are they popular?",
    "What is dollar-cost averaging in investing?",
    "What does it mean to diversify an investment portfolio?",
    "How do I write a professional resignation letter?",
    "What are some strategies for managing workplace stress?",
    "How do I give constructive feedback to a colleague?",
    "What is the difference between a trademark and a copyright?",
    "How does a VPN work and when should I use one?",
    "What is multi-factor authentication and how does it improve security?",
    "What is phishing and how do I recognize it?",
    "How do I create a strong password?",
    "What is the difference between HTTP and HTTPS?",
    "How does a relational database differ from a NoSQL database?",
    "What is the CAP theorem in distributed systems?",
    "Explain the concept of eventual consistency.",
    "What is a microservices architecture?",
    "What is the difference between synchronous and asynchronous programming?",
    "How does garbage collection work in modern programming languages?",
    "What is the time complexity of binary search?",
    "What is dynamic programming and when is it useful?",
    "How does a hash table work?",
    "What is a linked list and when would you use one over an array?",
    "What is the difference between a stack and a queue?",
    "Explain the concept of recursion with an example.",
    "What is the observer design pattern?",
    "What is dependency injection and why is it useful?",
    "What is SOLID in object-oriented programming?",
    "What are the main differences between SQL and Python for data analysis?",
    "How do you handle missing values in a dataset?",
    "What is feature engineering in machine learning?",
    "What is overfitting and how do you prevent it?",
    "What is cross-validation and why is it important?",
    "What is the bias-variance tradeoff?",
    "What is a confusion matrix and what does it tell you?",
    "What is precision and recall in classification?",
    "What is the F1 score and when should you use it?",
    "How does gradient descent work?",
    "What is the difference between batch and stochastic gradient descent?",
    "What is a convolutional neural network used for?",
    "What is transfer learning?",
    "What is natural language processing?",
    "How do large language models generate text?",
    "What is a transformer architecture?",
    "What is attention in the context of neural networks?",
    "What is the difference between generative and discriminative models?",
    "What is reinforcement learning?",
    "How does A/B testing work?",
    "What is statistical significance and p-value?",
    "What is the difference between correlation and causation?",
    "What is a hypothesis test?",
    "What is the central limit theorem?",
    "What is Bayes theorem and how is it applied?",
    "What is the difference between parametric and non-parametric tests?",
    "What is a confidence interval?",
    "How do you interpret a regression coefficient?",
    "What is multicollinearity and how does it affect regression?",
    "What is the difference between Type I and Type II errors?",
    "What is ANOVA and when is it used?",
    "How does logistic regression differ from linear regression?",
    "What are the assumptions of linear regression?",
    "What is the difference between population and sample in statistics?",
    "How do you calculate standard deviation?",
    "What is the difference between mean, median, and mode?",
    "What is a normal distribution?",
    "What is a Poisson distribution and when is it used?",
    "What are the differences between IPv4 and IPv6?",
    "What is a subnet mask?",
    "What is the OSI model in networking?",
    "What is a firewall and how does it work?",
    "What is load balancing in web infrastructure?",
    "What is a CDN and why is it useful?",
    "What is Docker and why is containerization useful?",
    "What is Kubernetes and what problem does it solve?",
    "What is CI/CD in software development?",
    "What is Git and how does version control work?",
    "What is the difference between git merge and git rebase?",
    "What is a pull request in a code review workflow?",
    "What is technical debt and how should it be managed?",
    "What is Agile software development?",
    "What is the difference between Scrum and Kanban?",
    "What is a sprint retrospective in Agile?",
    "What is the role of a product manager?",
    "What is user story mapping?",
    "What is the minimum viable product concept?",
    "What does it mean to have product-market fit?",
    "How does cloud computing differ from traditional data centers?",
    "What is serverless computing?",
    "What is Infrastructure as Code?",
    "What is the difference between IaaS, PaaS, and SaaS?",
    "What is data warehousing and how does it differ from a database?",
    "What is ETL in data engineering?",
    "What is a data lake and how does it differ from a data warehouse?",
    "What is Apache Spark and what is it used for?",
    "What is stream processing and how does it differ from batch processing?",
    "What is the difference between OLAP and OLTP?",
    "What is data governance and why does it matter?",
    "What is a data catalog?",
    "What is the principle of least privilege in security?",
    "What is role-based access control?",
    "What is SQL injection and how do you prevent it?",
    "What is cross-site scripting and how is it prevented?",
    "What is CSRF and how do tokens mitigate it?",
    "What is OAuth and how does it work?",
    "What is JWT and how is it used for authentication?",
    "What is the difference between authentication and authorization?",
    "What is end-to-end encryption?",
    "What is TLS and how does it secure web traffic?",
    "What is a digital certificate and how does PKI work?",
    "What is hashing and how does it differ from encryption?",
    "What is salting a password and why is it important?",
]

# ---------------------------------------------------------------------------
# LIGHT prompts — 1 to 3 PII items → pipeline produces 1–3 tags → label 1
# ---------------------------------------------------------------------------
LIGHT_PROMPTS: list[str] = [
    # --- Name only (→ 1 tag) ---
    "My name is John Smith. How do I reset my online banking password?",
    "I am Sarah Johnson. Can you explain what a Roth IRA is?",
    "My name is Carlos Ruiz. What documents do I need to open a bank account?",
    "I am Lisa Wang. What is the best way to build an emergency fund?",
    "My name is David Park. How do I dispute an incorrect charge on my bill?",
    "I am Emily Chen. What should I know before applying for a mortgage?",
    "My name is Kevin Brown. How does health insurance deductible work?",
    "I am Anna Lee. What is the difference between a checking and savings account?",
    "My name is James Wilson. How do I freeze my credit report?",
    "I am Priya Sharma. What is a 401k and how does it work?",
    "My name is Michael Torres. How do I apply for a student loan?",
    "I am Rachel Kim. What are the tax implications of freelance income?",
    "My name is Robert Nguyen. How do I report a lost debit card?",
    "I am Jennifer Lopez. What is the difference between a W-2 and 1099?",
    "My name is Thomas Clark. What happens if I miss a credit card payment?",
    "I am Susan Martinez. How do I set up automatic bill payments?",
    "My name is William Anderson. What is a balance transfer and is it a good idea?",
    "I am Patricia Jackson. How do I get my free annual credit report?",
    "My name is Daniel White. What is the penalty for early withdrawal from a 401k?",
    "I am Nancy Harris. How do I set up a college savings plan for my child?",

    # --- Email only (→ 1 tag) ---
    "Please send account details to jsmith@gmail.com. What are the steps to close an account?",
    "My contact email is sarah.j@outlook.com. How do I update my billing address?",
    "Reach me at carlos.ruiz@company.com for any follow-up. What is overdraft protection?",
    "My email is lwang@mail.com. Can you explain wire transfer fees?",
    "Send confirmation to dpark@work.com. What is the SWIFT code used for?",
    "My email address is emily.chen@corp.com. What does ACH transfer mean?",
    "Contact me at kevin.brown@biz.net. How long does a check take to clear?",

    # --- Phone only (→ 1 tag) ---
    "Call me at 312-555-0192 to discuss my account options.",
    "My phone number is 415-555-0199. What are the steps to dispute a charge?",
    "Please call 617-555-0132 if you need more information about my claim.",
    "I can be reached at 213-555-0177. How do I apply for overdraft protection?",
    "My contact number is 800-555-1234. What documents are needed for a loan application?",

    # --- Demographic only (→ 1 tag) ---
    "I am a 35-year-old male. What health screenings should I schedule this year?",
    "As a 28-year-old woman, what should my retirement savings look like?",
    "I am a 52-year-old Hispanic professional. What Medicare options are available to me?",
    "As a 22-year-old college graduate, what is the best starter credit card?",
    "I am a 45-year-old female. What are the recommended mammogram screening guidelines?",
    "As a 60-year-old man, what should I know about Social Security benefits?",
    "I am a Korean-American student. What tax forms do I need as a non-resident?",
    "As an Indian professional on an H-1B visa, can I contribute to a 401k?",
    "I am a 30-year-old Mexican-American. What homebuyer programs are available to me?",
    "As a Chinese international student on an F-1 visa, can I open a US bank account?",

    # --- Name + email (→ 2 tags) ---
    "I am John Smith, email jsmith@gmail.com. What are the steps to dispute a billing error?",
    "My name is Sarah Johnson, email sjohnson@mail.com. How do I request a credit limit increase?",
    "I am Carlos Ruiz, contact carlos@email.com. What is the process for wiring money internationally?",
    "My name is Lisa Wang, email lwang@corp.com. What is a HELOC and how does it work?",
    "I am David Park, email dpark@biz.net. How do I transfer funds between accounts?",
    "My name is Emily Chen, email echen@work.com. What is the difference between APR and APY?",
    "I am Kevin Brown, email kbrown@test.org. What is a credit utilization ratio?",
    "My name is Anna Lee, email anna.lee@outlook.com. How do I set up direct deposit?",
    "I am James Wilson, email jwilson@gmail.com. What documents are needed for a home refinance?",
    "My name is Priya Sharma, email psharma@company.com. How do I close a savings account?",
    "I am Michael Torres, email mtorres@mail.com. What is a money market account?",
    "My name is Rachel Kim, email rkim@corp.net. How do I enroll in online banking?",
    "I am Robert Nguyen, email rnguyen@biz.com. What is the process to dispute a credit report error?",
    "My name is Jennifer Lopez, email jlopez@test.org. How do I sign up for account alerts?",
    "I am Thomas Clark, email tclark@mail.com. What is the difference between a fixed and variable interest rate?",

    # --- Name + phone (→ 2 tags) ---
    "My name is Susan Martinez, phone 702-555-0166. How does a home equity loan work?",
    "I am William Anderson, phone 312-555-0177. What is the difference between a debit and prepaid card?",
    "My name is Patricia Jackson, phone 502-555-0188. How do I change my PIN?",
    "I am Daniel White, phone 415-555-0199. What is a routing number used for?",
    "My name is Nancy Harris, phone 617-555-0111. How do I add a beneficiary to my account?",
    "I am Christopher Lewis, phone 213-555-0122. What is the process to stop a payment?",
    "My name is Dorothy Robinson, phone 800-555-0133. How do I apply for a personal loan?",
    "I am Mark Walker, phone 404-555-0144. What is the difference between secured and unsecured debt?",
    "My name is Sandra Hall, phone 614-555-0155. How does a certificate of deposit work?",
    "I am Steven Young, phone 512-555-0166. What is the best way to save for a down payment?",

    # --- Name + demographic (→ 2 tags) ---
    "I am Sarah Johnson, a 32-year-old woman. What are the best investment options for my age?",
    "My name is Carlos Ruiz, a 45-year-old Hispanic male. What estate planning steps should I take?",
    "I am Lisa Wang, a Chinese-American professional. What tax treaties apply to my situation?",
    "My name is David Park, a 28-year-old male. What type of life insurance should I get?",
    "I am Emily Chen, a 50-year-old woman. What should I prioritize financially in the next decade?",
    "My name is Kevin Brown, a 22-year-old student. What credit card should I apply for first?",
    "I am Anna Lee, a 38-year-old Korean-American. What disability insurance options are available?",
    "My name is James Wilson, a 55-year-old male. When should I start taking Social Security?",
    "I am Priya Sharma, an Indian professional. How do I file taxes as a permanent resident?",
    "My name is Michael Torres, a 40-year-old Hispanic man. What are COBRA insurance options?",

    # --- Name + email + phone (→ 3 tags) ---
    "I am John Smith, email jsmith@gmail.com, phone 312-555-0192. How do I open a joint account?",
    "My name is Sarah Johnson, email sjohnson@mail.com, phone 415-555-0199. What are the steps to refinance my mortgage?",
    "I am Carlos Ruiz, email carlos@mail.com, phone 617-555-0132. How do I report identity theft?",
    "My name is Lisa Wang, email lwang@corp.com, phone 213-555-0177. Can you explain investment risk tolerance?",
    "I am David Park, email dpark@biz.net, phone 800-555-1234. How do I dispute a fraudulent transaction?",
    "My name is Emily Chen, email echen@work.com, phone 404-555-0155. What is a fiduciary advisor?",
    "I am Kevin Brown, email kbrown@test.org, phone 702-555-0166. How do I improve my credit score?",
    "My name is Anna Lee, email anna.lee@outlook.com, phone 502-555-0177. What is the process to roll over a 401k?",
    "I am James Wilson, email jwilson@gmail.com, phone 614-555-0188. How do I set up a trust?",
    "My name is Priya Sharma, email psharma@company.com, phone 512-555-0199. What are the steps to buy an I bond?",

    # --- Name + card (→ 2 tags) ---
    "My name is John Smith. My card 4111111111111111 was declined. Why might this happen?",
    "I am Sarah Johnson. My credit card 5425233430109903 shows an unfamiliar charge.",
    "My name is Carlos Ruiz. Card 4532015112830366 was used at a store I did not visit.",
    "I am Lisa Wang. I need to cancel card 4111111111111118. What are the steps?",
    "My name is David Park. My card 5425233430109903 expired last month. How do I request a new one?",

    # --- Name + account number (→ 2 tags) ---
    "My name is Emily Chen. Account 987654321 shows an unexpected fee. How do I dispute it?",
    "I am Kevin Brown. Account number 246813579 has been locked. How do I unlock it?",
    "My name is Anna Lee. I want to close account 778899001. What is the procedure?",
    "I am James Wilson. Account 112233445 was debited twice. How do I request a refund?",
    "My name is Priya Sharma. I need to link account 564738291 to my mobile banking app.",

    # --- Name + demographic + email (→ 3 tags) ---
    "I am a 34-year-old named John Smith, email jsmith@mail.com. What insurance plans suit my age group?",
    "My name is Sarah Johnson, a Korean-American, email sjohnson@corp.com. What dual-filing rules apply to me?",
    "I am Carlos Ruiz, a 28-year-old male, email carlos@biz.net. What starter investment accounts do you recommend?",
    "My name is Lisa Wang, a Chinese-American professional, email lwang@test.org. Can I open an IRA as a green card holder?",
    "I am David Park, a 45-year-old Indian-American, email dpark@work.com. What are catch-up contribution limits?",
]

# ---------------------------------------------------------------------------
# HEAVY prompts — 4+ PII items → pipeline produces 4+ tags → label 2
# ---------------------------------------------------------------------------
HEAVY_PROMPTS: list[str] = [
    "I am a 34-year-old Indian-American named John Smith, email jsmith@gmail.com, at 123 Main St Boston MA. My card 4111111111111111 was charged $350 without authorization.",
    "My name is Sarah Johnson, a 28-year-old Korean-American, phone 312-555-0192, at 456 Oak Ave Chicago IL. Transfer $2,500 from account 987654321.",
    "I am Carlos Ruiz, a 40-year-old Hispanic male, email carlos@mail.com, at 789 Pine Rd Austin TX. My card 5425233430109903 shows a $1,200 charge.",
    "My name is Lisa Wang, a 35-year-old Chinese-American, phone 415-555-0199, at 200 Elm St San Francisco CA. Account 246813579 is showing a $500 error.",
    "I am David Park, a 29-year-old Korean male, email dpark@work.com, at 100 River Lane Denver CO. My card 4532015112830366 was used for $750.",
    "My name is Emily Chen, a 32-year-old Asian-American woman, phone 617-555-0132, at 300 Birch Blvd Seattle WA. Please move $3,000 from account 334455667.",
    "I am Kevin Brown, a 45-year-old African-American male, email kbrown@test.org, at 500 Maple Dr Atlanta GA. My card 4111111111111118 was charged $420.",
    "My name is Anna Lee, a 38-year-old Korean-American woman, phone 213-555-0177, at 400 Cedar St Houston TX. Account 778899001 has a $650 discrepancy.",
    "I am James Wilson, a 52-year-old white male, email jwilson@company.com, at 600 Walnut Ave Phoenix AZ. Transfer $1,800 from card 5425233430109903.",
    "My name is Priya Sharma, a 27-year-old Indian woman, phone 800-555-1234, at 700 Spruce Rd Miami FL. My account 564738291 shows $2,200 due.",
    "I am Michael Torres, a 36-year-old Hispanic male, email mtorres@mail.com, at 800 Ash Blvd Dallas TX. Card 4532015112830366 charged $950.",
    "My name is Rachel Kim, a 31-year-old Korean-American, phone 312-555-7788, at 900 Poplar St Portland OR. Account 112233445 has $1,100 pending.",
    "I am Robert Nguyen, a 44-year-old Vietnamese-American male, email rnguyen@corp.com, at 150 Willow Dr Minneapolis MN. My card 4111111111111111 shows $280 charge.",
    "My name is Jennifer Lopez, a 33-year-old Hispanic woman, phone 404-555-0155, at 250 Chestnut Ave Nashville TN. Transfer $4,500 to account 667788990.",
    "I am Thomas Clark, a 48-year-old white male, email tclark@mail.com, at 350 Magnolia Rd Baltimore MD. Card 5425233430109903 used for $1,600.",
    "My name is Susan Martinez, a 39-year-old Hispanic woman, phone 702-555-0166, at 450 Sycamore St Las Vegas NV. Account 889900112 shows $750 error.",
    "I am William Anderson, a 55-year-old white male, email wanderson@biz.com, at 550 Cypress Blvd Sacramento CA. Card 4532015112830366 charged $530.",
    "My name is Patricia Jackson, a 41-year-old African-American woman, phone 502-555-0177, at 650 Dogwood Dr Louisville KY. Move $2,000 from account 334455669.",
    "I am Daniel White, a 26-year-old male, email dwhite@org.net, at 750 Hawthorn Ave Indianapolis IN. Card 4111111111111111 shows $390.",
    "My name is Nancy Harris, a 58-year-old white woman, phone 901-555-0188, at 850 Juniper Rd Memphis TN. Account 556677889 has $1,400 pending.",
    "I am Christopher Lewis, a 34-year-old African-American male, email clewis@mail.com, at 950 Laurel St Charlotte NC. Card 5425233430109903 charged $870.",
    "My name is Dorothy Robinson, a 62-year-old white woman, phone 602-555-0199, at 1050 Locust Blvd Tucson AZ. Account 778800123 shows $620.",
    "I am Mark Walker, a 37-year-old white male, email mwalker@test.com, at 1150 Mulberry Dr Albuquerque NM. Transfer $3,300 from card 4532015112830366.",
    "My name is Sandra Hall, a 43-year-old African-American woman, phone 208-555-0111, at 1250 Myrtle Ave Boise ID. Card 4111111111111111 charged $490.",
    "I am Steven Young, a 30-year-old Asian-American male, email syoung@company.org, at 1350 Olive Rd Reno NV. Account 990011223 has $1,700 discrepancy.",
    "My name is Karen Allen, a 49-year-old white woman, phone 304-555-0122, at 1450 Palm St Charleston WV. Card 5425233430109903 used for $820.",
    "I am Edward King, a 56-year-old white male, email eking@biz.net, at 1550 Peach Blvd Richmond VA. Move $2,800 from account 445566778.",
    "My name is Betty Wright, a 64-year-old African-American woman, phone 505-555-0133, at 1650 Plum Dr Santa Fe NM. Card 4532015112830366 shows $360.",
    "I am George Scott, a 47-year-old white male, email gscott@mail.org, at 1750 Walnut Rd Amarillo TX. Account 667788992 charged $940.",
    "My name is Sharon Green, a 53-year-old African-American woman, phone 406-555-0144, at 1850 Beech Ave Billings MT. Transfer $1,500 from card 4111111111111111.",
    "I am Kenneth Baker, a 38-year-old white male, email kbaker@corp.com, at 1950 Elm Blvd Missoula MT. Card 5425233430109903 charged $710.",
    "My name is Jessica Adams, a 25-year-old Hispanic woman, phone 605-555-0155, at 2050 Fir St Sioux Falls SD. Account 778899003 shows $2,100.",
    "I am Paul Nelson, a 42-year-old white male, email pnelson@test.net, at 2150 Hemlock Dr Rapid City SD. Move $5,000 from account 334455671.",
    "My name is Helen Carter, a 60-year-old white woman, phone 701-555-0166, at 2250 Holly Ave Fargo ND. Card 4532015112830366 shows $480.",
    "I am Andrew Mitchell, a 33-year-old white male, email amitchell@mail.com, at 2350 Ivy Rd Grand Forks ND. Account 556600789 has $1,300.",
    "My name is Deborah Perez, a 46-year-old Hispanic woman, phone 402-555-0177, at 2450 Jasmine St Omaha NE. Card 4111111111111111 used for $650.",
    "I am Joshua Roberts, a 29-year-old white male, email jroberts@biz.com, at 2550 Larch Blvd Lincoln NE. Transfer $3,700 from account 889922334.",
    "My name is Lisa Turner, a 36-year-old African-American woman, phone 515-555-0188, at 2650 Lilac Dr Des Moines IA. Card 5425233430109903 charged $890.",
    "I am Stephen Phillips, a 51-year-old white male, email sphillips@org.net, at 2750 Linden Ave Cedar Rapids IA. Account 112211334 shows $1,050.",
    "My name is Shirley Campbell, a 44-year-old African-American woman, phone 816-555-0199, at 2850 Locust Rd Kansas City MO. Move $2,600 from card 4532015112830366.",
    "I am Jose Parker, a 31-year-old Hispanic male, email jparker@mail.com, at 2950 Magnolia Blvd St Louis MO. Card 4111111111111111 charged $420.",
    "My name is Virginia Evans, a 57-year-old white woman, phone 414-555-0111, at 3050 Maple Ave Milwaukee WI. Account 667733445 has $1,800 error.",
    "I am Ryan Edwards, a 27-year-old white male, email redwards@corp.com, at 3150 Mast Dr Madison WI. Transfer $4,200 from account 778844556.",
    "My name is Carolyn Collins, a 48-year-old white woman, phone 704-555-0122, at 3250 Meadow St Greensboro NC. Card 5425233430109903 used for $730.",
    "I am Timothy Stewart, a 35-year-old African-American male, email tstewart@test.org, at 3350 Mimosa Blvd Raleigh NC. Account 223344556 shows $2,300.",
    "My name is Amy Sanchez, a 30-year-old Hispanic woman, phone 615-555-0133, at 3450 Moss Rd Knoxville TN. Card 4532015112830366 charged $510.",
    "I am Gregory Morris, a 43-year-old African-American male, email gmorris@mail.net, at 3550 Mulberry Ave Chattanooga TN. Move $1,900 from account 334466778.",
    "My name is Angela Rogers, a 39-year-old African-American woman, phone 205-555-0144, at 3650 Myrtle Blvd Birmingham AL. Card 4111111111111111 shows $640.",
    "I am Larry Reed, a 54-year-old white male, email lreed@biz.com, at 3750 Needle St Montgomery AL. Account 889911223 has $1,600 pending.",
    "My name is Melissa Cook, a 28-year-old white woman, phone 251-555-0155, at 3850 Nettle Dr Mobile AL. Transfer $2,100 from card 5425233430109903.",
    "I am Walter Morgan, a 61-year-old white male, email wmorgan@corp.org, at 3950 Oak Ave Jackson MS. Card 4532015112830366 charged $880.",
    "My name is Catherine Bell, a 37-year-old African-American woman, phone 501-555-0166, at 4050 Olive St Little Rock AR. Account 556699001 shows $1,200.",
    "I am Patrick Murphy, a 46-year-old white male, email pmurphy@mail.com, at 4150 Orchid Rd Fort Smith AR. Move $3,400 from account 667700112.",
    "My name is Janet Bailey, a 52-year-old white woman, phone 318-555-0177, at 4250 Palm Blvd Shreveport LA. Card 4111111111111111 used for $760.",
    "I am Harold Rivera, a 40-year-old Hispanic male, email hrivera@test.com, at 4350 Peach Dr New Orleans LA. Account 112233448 has $2,400 error.",
    "My name is Diane Cooper, a 33-year-old white woman, phone 662-555-0188, at 4450 Pear Ave Biloxi MS. Transfer $1,700 from card 5425233430109903.",
    "I am Jerry Richardson, a 58-year-old African-American male, email jrichardson@biz.net, at 4550 Pine Blvd Gulfport MS. Card 4532015112830366 charged $920.",
    "My name is Gloria Cox, a 45-year-old African-American woman, phone 352-555-0199, at 4650 Plum St Gainesville FL. Account 334455680 shows $1,100.",
    "I am Randy Howard, a 36-year-old white male, email rhoward@corp.com, at 4750 Poplar Dr Tampa FL. Move $5,500 from account 445566791.",
    "My name is Martha Ward, a 63-year-old white woman, phone 904-555-0111, at 4850 Redbud Ave Jacksonville FL. Card 4111111111111111 charged $430.",
    "I am Russell Torres, a 29-year-old Hispanic male, email rtorres@mail.org, at 4950 Redwood Rd Orlando FL. Account 667788995 has $1,900 pending.",
    "My name is Evelyn Peterson, a 55-year-old white woman, phone 843-555-0122, at 5050 Rose Blvd Charleston SC. Transfer $2,800 from card 5425233430109903.",
    "I am Carl Gray, a 41-year-old African-American male, email cgray@test.net, at 5150 Rosemary St Columbia SC. Card 4532015112830366 used for $680.",
    "My name is Janet James, a 34-year-old white woman, phone 828-555-0133, at 5250 Rue Dr Asheville NC. Account 778833445 shows $2,600.",
    "I am Raymond Watson, a 49-year-old African-American male, email rwatson@mail.com, at 5350 Sand Ave Savannah GA. Move $1,300 from account 889944556.",
    "My name is Ruby Brooks, a 38-year-old African-American woman, phone 770-555-0144, at 5450 Sequoia Blvd Macon GA. Card 4111111111111111 charged $570.",
    "I am Roger Kelly, a 53-year-old white male, email rkelly@biz.com, at 5550 Spruce St Augusta GA. Account 334400112 has $1,500 discrepancy.",
    "My name is Judith Sanders, a 42-year-old white woman, phone 478-555-0155, at 5650 Sycamore Dr Columbus GA. Transfer $4,000 from card 5425233430109903.",
    "I am Joe Price, a 31-year-old Hispanic male, email jprice@corp.org, at 5750 Tamarack Ave Athens GA. Card 4532015112830366 shows $840.",
    "My name is Jean Bennett, a 60-year-old white woman, phone 334-555-0166, at 5850 Thistle Rd Huntsville AL. Account 556677901 has $1,700.",
    "I am Billy Wood, a 47-year-old white male, email bwood@mail.net, at 5950 Tulip Blvd Decatur AL. Move $2,300 from account 667788902.",
    "My name is Theresa Barnes, a 35-year-old African-American woman, phone 256-555-0177, at 6050 Verbena St Anniston AL. Card 4111111111111111 used for $690.",
    "I am Austin Ross, a 26-year-old white male, email aross@test.com, at 6150 Violet Dr Dothan AL. Account 112200334 shows $1,400.",
    "My name is Christina Henderson, a 44-year-old Hispanic woman, phone 985-555-0188, at 6250 Walnut Ave Baton Rouge LA. Transfer $3,600 from card 5425233430109903.",
    "I am Willie Coleman, a 57-year-old African-American male, email wcoleman@biz.org, at 6350 Willow Rd Lafayette LA. Card 4532015112830366 charged $780.",
    "My name is Mildred Jenkins, a 65-year-old African-American woman, phone 979-555-0199, at 6450 Wisteria Blvd Beaumont TX. Account 334466790 has $2,200.",
    "I am Eugene Patterson, a 39-year-old African-American male, email epatterson@corp.com, at 6550 Yarrow St Lubbock TX. Move $1,600 from account 445577891.",
    "My name is Wanda Alexander, a 32-year-old African-American woman, phone 806-555-0111, at 6650 Zinnia Dr Amarillo TX. Card 4111111111111111 charged $510.",
    "I am Nathan Russell, a 28-year-old white male, email nrussell@mail.com, at 6750 Acorn Ave El Paso TX. Account 667799002 shows $1,300.",
    "My name is Frances Griffin, a 46-year-old African-American woman, phone 432-555-0122, at 6850 Acacia Blvd Midland TX. Transfer $2,700 from card 5425233430109903.",
    "I am Ryan Hughes, a 37-year-old white male, email rhughes@test.net, at 6950 Alder Rd Odessa TX. Card 4532015112830366 used for $960.",
    "My name is Marie Foster, a 51-year-old Hispanic woman, phone 915-555-0133, at 7050 Almond St San Antonio TX. Account 778855446 has $1,800.",
    "I am Frank Powell, a 43-year-old white male, email fpowell@biz.com, at 7150 Aspen Blvd Corpus Christi TX. Move $4,100 from account 889966557.",
    "My name is Julie Long, a 29-year-old white woman, phone 361-555-0144, at 7250 Aster Ave Laredo TX. Card 4111111111111111 charged $640.",
    "I am Bruce Patterson, a 55-year-old African-American male, email bpatterson@corp.org, at 7350 Bamboo Dr McAllen TX. Account 112233459 shows $2,000.",
    "My name is Lauren Coleman, a 33-year-old African-American woman, phone 956-555-0155, at 7450 Bark Blvd Brownsville TX. Transfer $1,400 from card 5425233430109903.",
    "I am Scott Simmons, a 40-year-old white male, email ssimmons@mail.net, at 7550 Bayberry St Fort Worth TX. Card 4532015112830366 shows $720.",
    "My name is Victoria Foster, a 48-year-old white woman, phone 817-555-0166, at 7650 Birch Ave Arlington TX. Account 334477889 has $1,100.",
    "I am Jack Bryant, a 27-year-old white male, email jbryant@test.com, at 7750 Black Rd Plano TX. Move $5,800 from account 445588990.",
    "My name is Hannah Diaz, a 36-year-old Hispanic woman, phone 972-555-0177, at 7850 Blackberry Blvd Garland TX. Card 4111111111111111 charged $390.",
    "I am Keith Alexander, a 52-year-old African-American male, email kalexander@biz.org, at 7950 Blossom St Mesquite TX. Account 667711223 shows $2,500.",
    "My name is Gloria Graham, a 44-year-old African-American woman, phone 469-555-0188, at 8050 Blue Rd Irving TX. Transfer $3,100 from card 5425233430109903.",
    "I am Roy Gonzalez, a 31-year-old Hispanic male, email rgonzalez@corp.com, at 8150 Bluebell Ave Grand Prairie TX. Card 4532015112830366 used for $830.",
    "My name is Kathleen Wood, a 59-year-old white woman, phone 214-555-0199, at 8250 Bluebird Blvd Denton TX. Account 778800334 has $1,600.",
    "I am Donald Tucker, a 46-year-old white male, email dtucker@mail.com, at 8350 Bluegrass Dr Waco TX. Move $2,500 from account 889911445.",
    "My name is Cynthia Owens, a 38-year-old African-American woman, phone 254-555-0111, at 8450 Boxwood St Killeen TX. Card 4111111111111111 charged $460.",
    "I am Douglas Lane, a 53-year-old white male, email dlane@test.net, at 8550 Brier Ave Temple TX. Account 556644778 shows $1,900.",
    "My name is Janice Dean, a 41-year-old African-American woman, phone 254-555-0122, at 8650 Bristle Rd Round Rock TX. Transfer $1,200 from card 5425233430109903.",
    "I am Anthony Ray, a 34-year-old African-American male, email aray@biz.com, at 8750 Buckwheat Blvd Abilene TX. Card 4532015112830366 charged $870.",
    "My name is Cheryl Arnold, a 50-year-old white woman, phone 325-555-0133, at 8850 Bur Oak Rd San Angelo TX. Account 667755889 has $2,100.",
    "I am Samuel Hawkins, a 43-year-old African-American male, email shawkins@corp.org, at 8950 Bur Ave Tyler TX. Move $3,900 from account 778866990.",
    "My name is Brenda Stone, a 56-year-old white woman, phone 903-555-0144, at 9050 Bush Blvd Longview TX. Card 4111111111111111 used for $580.",
    "I am Jonathan Morales, a 30-year-old Hispanic male, email jmorales@mail.net, at 9150 Buttonwood St Texarkana TX. Account 112244556 shows $1,300.",
    "My name is Denise Torres, a 37-year-old Hispanic woman, phone 903-555-0155, at 9250 Cactus Dr Beaumont TX. Transfer $4,600 from card 5425233430109903.",
    "I am Jerry Patterson, a 48-year-old African-American male, email jmpatterson@test.com, at 9350 Camellia Blvd Galveston TX. Card 4532015112830366 shows $790.",
    "My name is Linda Carter, a 35-year-old Hispanic woman, phone 409-555-0166, at 9450 Canyon Ave Pasadena TX. Account 334488889 has $1,400.",
    "I am Matthew Murphy, a 42-year-old white male, email mmurphy@biz.org, at 9550 Catalpa Rd The Woodlands TX. Move $2,200 from account 445599990.",
    "My name is Robin Bailey, a 29-year-old white woman, phone 281-555-0177, at 9650 Cedar Ave Sugar Land TX. Card 4111111111111111 charged $520.",
    "I am Nicholas Price, a 55-year-old white male, email nprice@corp.com, at 9750 Cedar Blvd Pearland TX. Account 667722334 shows $1,700.",
    "My name is Carol Flores, a 40-year-old Hispanic woman, phone 346-555-0188, at 9850 Cedarwood St Baytown TX. Transfer $3,300 from card 5425233430109903.",
    "I am Peter Gray, a 33-year-old white male, email pgray@mail.com, at 9950 Chestnut Dr League City TX. Card 4532015112830366 used for $650.",
    "My name is Maria Wallace, a 47-year-old Hispanic woman, phone 281-555-0199, at 10050 Elm Blvd Friendswood TX. Account 778811445 has $2,400.",
    "I am Henry Adams, a 61-year-old white male, email hadams@test.net, at 10150 Oak Rd Missouri City TX. Move $1,800 from account 889922556.",
    "My name is Teresa Banks, a 36-year-old African-American woman, phone 832-555-0111, at 10250 Clover Ave Conroe TX. Card 4111111111111111 charged $410.",
    "I am Arthur Watson, a 52-year-old African-American male, email awatson@biz.com, at 10350 Cloverleaf Blvd Katy TX. Account 334455791 shows $1,500.",
    "My name is Maria Henderson, a 44-year-old Hispanic woman, phone 281-555-0122, at 10450 Cobblestone St Spring TX. Transfer $2,900 from card 5425233430109903.",
    "I am Raymond Hall, a 39-year-old African-American male, email rhall@corp.org, at 10550 Coconut Rd Humble TX. Card 4532015112830366 charged $770.",
    "My name is Amanda Simmons, a 31-year-old white woman, phone 281-555-0133, at 10650 Columbine Ave Kingwood TX. Account 556678002 has $1,200.",
    "I am Jerry Sanders, a 57-year-old white male, email jsanders@mail.net, at 10750 Copperleaf Blvd Tomball TX. Move $4,700 from account 667789113.",
    "My name is Rebecca Long, a 34-year-old white woman, phone 281-555-0144, at 10850 Coral Rd Cypress TX. Card 4111111111111111 used for $560.",
    "I am Willie James, a 49-year-old African-American male, email wjames@test.com, at 10950 Coralbells Blvd Rosenberg TX. Account 778833556 shows $1,900.",
    "My name is Patricia Hayes, a 42-year-old African-American woman, phone 281-555-0155, at 11050 Cornflower St Richmond TX. Transfer $2,100 from card 5425233430109903.",
    "I am Aaron Dixon, a 28-year-old African-American male, email adixon@biz.org, at 11150 Cosmos Ave Webster TX. Card 4532015112830366 charged $830.",
    "My name is Lori Barnes, a 53-year-old white woman, phone 281-555-0166, at 11250 Cottonwood Dr Alvin TX. Account 334466901 has $1,600.",
    "I am Raymond Turner, a 37-year-old African-American male, email rturner@corp.com, at 11350 Crabapple Blvd Deer Park TX. Move $3,500 from account 445577012.",
    "My name is Susan Powell, a 45-year-old white woman, phone 832-555-0177, at 11450 Cranberry Rd Channelview TX. Card 4111111111111111 charged $490.",
    "I am Joshua Coleman, a 32-year-old African-American male, email jcoleman@mail.com, at 11550 Cranesbill Blvd Clute TX. Account 667700223 shows $2,200.",
    "My name is Jennifer Ross, a 38-year-old white woman, phone 979-555-0188, at 11650 Creosote St Lake Jackson TX. Transfer $1,500 from card 5425233430109903.",
    "I am Henry Stewart, a 60-year-old white male, email hstewart@test.net, at 11750 Crepe Dr Angleton TX. Card 4532015112830366 used for $760.",
    "My name is Sandra Hughes, a 43-year-old Hispanic woman, phone 979-555-0199, at 11850 Crocus Ave Brazoria TX. Account 112299667 has $1,300.",
    "I am Joe Foster, a 55-year-old white male, email jfoster@biz.com, at 11950 Crow Blvd Clute TX. Move $5,200 from account 223300778.",
    "My name is Linda Graham, a 30-year-old African-American woman, phone 979-555-0111, at 12050 Crown Rd Freeport TX. Card 4111111111111111 charged $620.",
    "I am Thomas Myers, a 47-year-old white male, email tmyers@corp.org, at 12150 Cypress Blvd Lake Charles LA. Account 334411889 shows $1,800.",
    "My name is Betty Shaw, a 58-year-old white woman, phone 337-555-0122, at 12250 Daffodil Ave Lake Charles LA. Transfer $2,600 from card 5425233430109903.",
    "I am Kenneth Woods, a 36-year-old African-American male, email kwoods@mail.net, at 12350 Daisy Dr Sulphur LA. Card 4532015112830366 charged $900.",
    "My name is Carol Mason, a 51-year-old white woman, phone 337-555-0133, at 12450 Dandelion Blvd Westlake LA. Account 556622990 has $1,700.",
    "I am Aaron Hall, a 29-year-old African-American male, email ahall@test.com, at 12550 Deadwood Rd Vinton LA. Move $1,900 from account 667733001.",
    "My name is Christine Butler, a 44-year-old white woman, phone 337-555-0144, at 12650 Delphinium St Jennings LA. Card 4111111111111111 used for $540.",
    "I am Gary Roberts, a 39-year-old white male, email groberts@biz.org, at 12750 Douglas Rd Crowley LA. Account 778844112 shows $2,000.",
    "My name is Kathleen Collins, a 56-year-old white woman, phone 337-555-0155, at 12850 Dove Blvd Opelousas LA. Transfer $3,800 from card 5425233430109903.",
    "I am Dennis Richardson, a 48-year-old white male, email drichardson@corp.com, at 12950 Dragon Rd New Iberia LA. Card 4532015112830366 charged $810.",
    "My name is Sharon Lewis, a 33-year-old African-American woman, phone 337-555-0166, at 13050 Drift Ave Morgan City LA. Account 889955334 shows $1,500.",
    "I am Eric Washington, a 41-year-old African-American male, email ewashington@mail.net, at 13150 Dune Blvd Houma LA. Move $2,700 from account 112266778.",
    "My name is Patricia Martin, a 52-year-old Hispanic woman, phone 985-555-0177, at 13250 Dutch Rd Thibodaux LA. Card 4111111111111111 charged $670.",
    "I am Larry Thompson, a 37-year-old African-American male, email lthompson@test.com, at 13350 Eagle Dr Slidell LA. Account 334477890 shows $1,100.",
    "My name is Diane Garcia, a 46-year-old Hispanic woman, phone 985-555-0188, at 13450 Elder Blvd Covington LA. Transfer $4,300 from card 5425233430109903.",
    "I am Willie Martinez, a 59-year-old Hispanic male, email wmartinez@biz.org, at 13550 Elderberry St Mandeville LA. Card 4532015112830366 used for $940.",
]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_ollama_embedding(text: str, retries: int = 3) -> list[float] | None:
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_MODEL, "prompt": text},
                timeout=60.0,
            )
            resp.raise_for_status()
            emb = resp.json().get("embedding")
            if emb:
                return emb
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2.0)
            else:
                print(f"\n  WARNING: embedding failed: {e}")
    return None


def load_cache() -> dict[str, list[float]]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict[str, list[float]]) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

# 127 base prompts: 40 clean + 50 light + 37 heavy
BASE_PROMPTS: list[str] = CLEAN_PROMPTS[:40] + LIGHT_PROMPTS[:50] + HEAVY_PROMPTS[:37]


def build_dataset() -> tuple[list[str], list[float]]:
    """
    Paired approach: for each of the 127 base prompts, add:
      - The original prompt    → target 0.0  (no redaction tags)
      - The pipeline output    → target = redacted_chars / total_chars

    The regression target directly encodes redaction density, so the probe
    learns "more redaction → higher score" as its training objective.
    """
    texts:   list[str]   = []
    targets: list[float] = []
    n = len(BASE_PROMPTS)
    n_pairs = 0
    print(f"  Running pipeline on {n} base prompts...")
    for i, prompt in enumerate(BASE_PROMPTS):
        try:
            redacted = sequential_redaction_pipeline(prompt, MODULE_SETTINGS)
        except Exception as e:
            print(f"\n  WARNING [{i+1}]: {e}")
            redacted = prompt

        texts.append(prompt)
        targets.append(0.0)

        ratio = redaction_ratio(redacted)
        texts.append(redacted)
        targets.append(ratio)
        if redacted != prompt:
            n_pairs += 1

        print(f"  [{i+1}/{n}] tags={count_tags(redacted)} ratio={ratio:.3f} pairs={n_pairs}", end="\r")
    print()
    return texts, targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Step 1: Checking Ollama ===")
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        print(f"  Ollama reachable. Models: {models}")
        if not any(m.startswith(OLLAMA_MODEL.split(":")[0]) for m in models):
            print(f"  ERROR: '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
            return
        print(f"  '{OLLAMA_MODEL}' confirmed.")
    except Exception as e:
        print(f"  ERROR: {e}\n  Run: ollama serve")
        return

    print(f"\n=== Step 2: Building dataset ({len(BASE_PROMPTS)} base prompts, paired) ===")
    texts, targets = build_dataset()

    import numpy as np
    y_arr = np.array(targets)
    print(f"  Total examples:  {len(texts)}  (original + pipeline pairs)")
    print(f"  Target=0.0 (no redaction): {(y_arr == 0.0).sum()}")
    print(f"  Target>0.0 (some redaction): {(y_arr > 0.0).sum()}")
    print(f"  Target range: [{y_arr.min():.3f}, {y_arr.max():.3f}]")

    print("\n  Sample:")
    for idx in [0, len(texts)//2, len(texts)-1]:
        print(f"  [{idx}] ratio={targets[idx]:.3f}  tags={count_tags(texts[idx])}  {texts[idx][:70]}")

    print("\n=== Step 3: Embedding (cached per prompt) ===")
    cache = load_cache()
    new_texts = [t for t in texts if t not in cache]
    print(f"  Cache hit: {len(texts) - len(new_texts)}   Need to embed: {len(new_texts)}")

    if new_texts:
        t0 = time.time()
        for i, text in enumerate(new_texts):
            emb = get_ollama_embedding(text)
            if emb is not None:
                cache[text] = emb
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            remaining = avg * (len(new_texts) - i - 1)
            print(f"  [{i+1}/{len(new_texts)}]  elapsed={elapsed:.0f}s  est_remaining={remaining:.0f}s", end="\r")
        print(f"\n  Embedded {len(new_texts)} new prompts.")
        save_cache(cache)
        print(f"  Cache saved: {CACHE_FILE}")

    embeddings = [cache[t] for t in texts if t in cache]
    valid_targets = [targets[i] for i, t in enumerate(texts) if t in cache]
    if len(embeddings) < 50:
        print("  ERROR: Too few embeddings. Aborting.")
        return

    print(f"\n=== Step 4: Training Ridge regression (redaction ratio) ===")
    import torch
    X = np.array(embeddings, dtype=np.float32)
    y = np.array(valid_targets, dtype=np.float32)
    emb_dim = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}   Test: {len(X_test)}   dim={emb_dim}")

    clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"  MAE: {mae:.4f}   R²: {r2:.4f}")

    w_t = torch.tensor(clf.coef_, dtype=torch.float32)        # [4096]
    b_t = torch.tensor(clf.intercept_, dtype=torch.float32)   # scalar

    print("\n=== Step 5: Spot-check gradation ===")
    spot_checks = [
        ("No PII (clean)",              "What is the best way to learn Python programming?"),
        ("1 tag — name only",           "My name is [REDACTED NAME]. I need help with my account."),
        ("2 tags — name + email",       "I am [REDACTED NAME], email [REDACTED EMAIL]. I have a billing question."),
        ("3 tags — name + email + phone", "[REDACTED NAME] [REDACTED EMAIL] [REDACTED PHONE]. Please contact me."),
        ("4 tags — name+email+loc+card","I am [REDACTED NAME], [REDACTED EMAIL], at [REDACTED LOCATION]. Card [REDACTED CARD]."),
        ("5 tags — heavy",              "[REDACTED NAME] [REDACTED EMAIL] [REDACTED LOCATION] [REDACTED CARD] [REDACTED VALUE]."),
        ("6 tags — fully redacted",     "[REDACTED NAME] [REDACTED DEMOGRAPHIC] [REDACTED LOCATION] [REDACTED CARD] [REDACTED ACCOUNT] [REDACTED VALUE]."),
    ]
    for label, text in spot_checks:
        emb = get_ollama_embedding(text)
        if emb is None:
            print(f"  [{label}]  SKIPPED"); continue
        emb_t = torch.tensor(emb, dtype=torch.float32)
        raw   = float(torch.dot(w_t, emb_t) + b_t)
        score = max(0.0, min(1.0, raw))
        print(f"  [{label}]  tags={count_tags(text)}  ratio={redaction_ratio(text):.3f}  score={score:.4f}")

    print("\n=== Step 6: Saving probe ===")
    probe_data = {
        "w":          w_t,           # [4096]
        "b":          b_t,           # scalar
        "n_classes":  2,
        "layer":      32,
        "backbone":   OLLAMA_MODEL,
        "probe_type":  "regression",
        "target":      "redaction_ratio = redacted_chars / total_chars",
        "model_type":  "Ridge regression (paired training)",
        "scoring":     "score = clip(w · embedding + b, 0, 1)",
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "mae":        round(float(mae), 4),
        "r2":         round(float(r2),  4),
        "emb_dim":    emb_dim,
    }
    os.makedirs(os.path.dirname(OUTPUT_PROBE), exist_ok=True)
    torch.save(probe_data, OUTPUT_PROBE)
    print(f"  Saved: {OUTPUT_PROBE}")
    print(f"  w shape: {w_t.shape}   b shape: {b_t.shape}")
    print(f"  MAE: {mae:.4f}   R²: {r2:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
