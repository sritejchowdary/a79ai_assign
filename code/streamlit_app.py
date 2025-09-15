import os
import time
import constant
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_calendar import calendar
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

df = pd.read_csv(constant.csv_file_path)

# load Chroma

def get_chroma():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=constant.AZURE_OPENAI_API_EMBEDDING_DEPLOYMENT_NAME,
        openai_api_version=constant.OPENAI_API_VERSION,
        azure_endpoint=constant.AZURE_OPENAI_ENDPOINT,
        api_key=constant.AZURE_OPENAI_KEY,
    )
    return Chroma(
        persist_directory=constant.persist_directory,
        embedding_function=embeddings    )

# -------------------------------------------------
# Agents
# -------------------------------------------------
def qa_agent(question: str, db):
    docs = db.similarity_search(question, k=3)
    if not docs:
        return "No relevant information found."

    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    prompt_template = ChatPromptTemplate.from_template(constant.PROMPT_TEMPLATE_QA)
    prompt = prompt_template.format(vector_context=context_text, question=question)

    model = AzureChatOpenAI(
        azure_deployment=constant.AZURE_OPENAI_DEPLOYMENT_NAME,
        openai_api_version=constant.OPENAI_API_VERSION,
        azure_endpoint=constant.AZURE_OPENAI_ENDPOINT,
        api_key=constant.AZURE_OPENAI_KEY
    )

    response = model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def insight_agent(question: str, db):
    docs = db.similarity_search(question, k=5)
    if not docs:
        return "No related insights found."

    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    prompt_template = ChatPromptTemplate.from_template(constant.PROMPT_TEMPLATE_INSIGHT)
    prompt = prompt_template.format(vector_context=context_text, question=question)

    model = AzureChatOpenAI(
        azure_deployment=constant.AZURE_OPENAI_DEPLOYMENT_NAME,
        openai_api_version=constant.OPENAI_API_VERSION,
        azure_endpoint=constant.AZURE_OPENAI_ENDPOINT,
        api_key=constant.AZURE_OPENAI_KEY
    )

    response = model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

db = get_chroma()

st.set_page_config(layout="wide")

# # HCP Selection
# Page navigation
page = st.query_params.get("page", "Chatbot")

if page == "Chatbot":
    col_left, col_right = st.columns([2, 1])
    st.set_page_config(page_title="HCP Assistant Chatbot")
    with col_left:

        st.markdown("## 💬 HCP Assistant Chatbot")

        hcp_options = df['hcp_name'] + ' (' + df['hcp_id'] + ')'
        if "selected_hcp" not in st.session_state:
            st.session_state.selected_hcp = hcp_options[0] if not hcp_options.empty else None
        st.markdown("<div style='text-align:left;margin-top:20px;'>", unsafe_allow_html=True)
        hcp_search = st.text_input('Search HCPs by name or ID:', '', key='hcp_search')
        filtered_hcp_options = [hcp for hcp in hcp_options if hcp_search.lower() in hcp.lower()]
        if not filtered_hcp_options:
            filtered_hcp_options = hcp_options
        selected_hcp = st.selectbox('Choose an HCP:', filtered_hcp_options, index=0 if st.session_state.selected_hcp is None or st.session_state.selected_hcp not in filtered_hcp_options else filtered_hcp_options.index(st.session_state.selected_hcp), key="hcp_select")
        st.session_state.selected_hcp = selected_hcp
        st.markdown("<div style='text-align:left;margin-top:20px;'>", unsafe_allow_html=True)
        selected_row = df[df['hcp_name'] + ' (' + df['hcp_id'] + ')' == st.session_state.selected_hcp].iloc[0]
        
        chat_container = st.container()
        with chat_container:
            if "messages" not in st.session_state:
                st.session_state.messages = []
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            if prompt := st.chat_input("Ask me about HCPs, patients, claims..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    qa_response = qa_agent(prompt, db)
                    insight_response = insight_agent(prompt, db)
                    answer = f"**Q&A Agent:** {qa_response}\n\n**Insight Agent:** {insight_response}"
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        st.markdown('<div style="text-align:left;margin-top:20px;"><a href="?page=Dashboard"><button style="background-color:#2196f3;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;">Dashboard</button></a></div>', unsafe_allow_html=True)

# Calendar and Notes
    with col_right:
        st.markdown("### Calendar")
        # Prepare call date data
        df_calls = df.copy()
        df_calls['call_date'] = pd.to_datetime(df_calls['call_date'], errors='coerce')
        df_calls = df_calls.dropna(subset=['call_date'])

        # Prepare events for streamlit-calendar
        events = []
        for _, row in df_calls.iterrows():
            events.append({
                "title": f"Call: {row['hcp_name']}",
                "start": row['call_date'].strftime('%Y-%m-%d'),
                "end": row['call_date'].strftime('%Y-%m-%d'),
                "allDay": True,
                "tooltip": row['hcp_name']
            })

        calendar_options = {
            "initialView": "dayGridMonth",
            "headerToolbar": {
                "left": "prev,next today",
                "center": "title",
                "right": "dayGridMonth,timeGridWeek,timeGridDay"
            },
            "eventMouseEnter": True,
        }

        calendar(
            events=events,
            options=calendar_options,
            custom_css=".fc-event-title { font-size: 1rem; }",
            key="calendar1"
        )
        st.markdown('---')
        st.markdown(f'### Insights Based on Previous Conversations with HCP {selected_row["hcp_name"]}')
        notes_text = ''
        if 'notes' in df.columns:
            notes_text = str(selected_row['notes']) if not pd.isna(selected_row['notes']) else 'No notes available.'
        else:
            notes_text = 'No notes column found in CSV.'
        st.markdown(f'<div style="background:#222;padding:12px;border-radius:8px;min-height:80px;">{notes_text}</div>', unsafe_allow_html=True)

# Page 2: Dashboard
elif page == "Dashboard":
    st.set_page_config(page_title="HCP Dashboard")
    st.markdown('<div style="text-align:right;margin-bottom:20px;"><a href="?page=Chatbot"><button style="background-color:#2196f3;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;">Chatbot</button></a></div>', unsafe_allow_html=True)
    st.markdown("## 📊 HCP Dashboard")

    hcp_options = df['hcp_name'] + ' (' + df['hcp_id'] + ')'
    if "selected_hcp" not in st.session_state:
        st.session_state.selected_hcp = hcp_options[0] if not hcp_options.empty else None
    st.markdown("<div style='text-align:center;margin-bottom:20px;'>", unsafe_allow_html=True)
    hcp_search = st.text_input('Search HCPs by name or ID:', '', key='hcp_search')
    filtered_hcp_options = [hcp for hcp in hcp_options if hcp_search.lower() in hcp.lower()]
    if not filtered_hcp_options:
        filtered_hcp_options = hcp_options
    selected_hcp = st.selectbox('Choose an HCP:', filtered_hcp_options, index=0 if st.session_state.selected_hcp is None or st.session_state.selected_hcp not in filtered_hcp_options else filtered_hcp_options.index(st.session_state.selected_hcp), key="hcp_select")
    st.session_state.selected_hcp = selected_hcp
    st.markdown("</div>", unsafe_allow_html=True)
    selected_row = df[df['hcp_name'] + ' (' + df['hcp_id'] + ')' == st.session_state.selected_hcp].iloc[0]

    st.markdown('<hr style="margin-top:0;margin-bottom:16px;border:0;border-top:1px solid #eee;"/>', unsafe_allow_html=True)
    header_col1, header_col2, header_col3, header_col4 = st.columns(4)

    with header_col1:
        st.markdown("""
            <div style='text-align:center;'>
                <img src='https://cdn-icons-png.flaticon.com/512/387/387561.png' width='90'/>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:32px;font-weight:700;margin-bottom:2px;text-align:center'>{selected_row['hcp_name']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center'><span style='color:#2196f3;font-size:20px'><b>NPI:</b></span> <span style='font-size:18px'>{selected_row['hcp_id']}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center'><span style='color:#2196f3;font-size:20px'><b>Address:</b></span> <span style='font-size:18px'>{selected_row['hcp_address']}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center'><span style='color:#2196f3;font-size:20px'><b>Specialty:</b></span> <span style='font-size:18px'>{selected_row['hcp_specialty']}</span></div>", unsafe_allow_html=True)

    with header_col2:
        st.markdown("""
            <div style='text-align:center;'>
                <img src='https://cdn-icons-png.flaticon.com/512/1430/1430453.png' width='70'/>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:#2196f3;font-size:22px;font-weight:600;">Total Patients</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;font-size:36px;font-weight:700'>{int(selected_row['total_patients']):,}</div>", unsafe_allow_html=True)

    with header_col3:
        st.markdown("""
            <div style='text-align:center;'>
                <img src='https://cdn-icons-png.flaticon.com/512/2770/2770435.png' width='40'/>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:#2196f3;font-size:22px;font-weight:600;">Diagnosed Patients</span></div>', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;font-size:36px;font-weight:700'>{int(selected_row['diagnosed_patients']):,}</div>", unsafe_allow_html=True)

    with header_col4:
        st.markdown("""
            <div style='text-align:center;'>
                <img src='https://cdn-icons-png.flaticon.com/512/16498/16498868.png' width='40'/>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:#2196f3;font-size:22px;font-weight:600;">Treated Patients</span></div>', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;font-size:36px;font-weight:700'>{int(selected_row['treated_patients']):,}</div>", unsafe_allow_html=True)

    st.markdown('<hr style="margin-top:16px;margin-bottom:0;border:0;border-top:1px solid #eee;"/>', unsafe_allow_html=True)

    # Gender Distribution
    gender_labels = ['Male', 'Female', 'Unknown']
    claims_gender = [int(selected_row['male_claims']), int(selected_row['female_claims']), int(selected_row['unknown_claims'])]
    patients_gender = [int(selected_row['male_patients']), int(selected_row['female_patients']), int(selected_row['unknown_patients'])]

    st.markdown('### Patient Count')
    col_patient1, col_patient2 = st.columns([1,2])
    with col_patient1:
        st.markdown('<div style="font-size:22px;color:#fff;">Overall Patient Count</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:48px;font-weight:500;color:#fff;">{int(selected_row["total_patients"]):,}</div>', unsafe_allow_html=True)
    with col_patient2:
        st.markdown('<div style="font-size:22px;color:#fff;">Patient Count by Gender</div>', unsafe_allow_html=True)
        fig_patients_gender = go.Figure(data=[go.Pie(labels=gender_labels, values=patients_gender, hole=.5)])
        fig_patients_gender.update_layout(title_text='', showlegend=True)
        st.plotly_chart(fig_patients_gender, use_container_width=True)

    st.markdown('<div style="font-size:22px;color:#fff;">Patients Count by Months</div>', unsafe_allow_html=True)
    patient_month_cols = [col for col in df.columns if col.startswith('patient_count_2024-')]
    patient_months = [col.split('_')[-1] for col in patient_month_cols]
    patient_month_values = [int(selected_row[col]) for col in patient_month_cols]
    fig_patient_line = go.Figure()
    fig_patient_line.add_trace(go.Scatter(x=patient_months, y=patient_month_values, mode='lines+markers', name='Patients'))
    fig_patient_line.update_layout(title_text='', xaxis_title='Month', yaxis_title='Count')
    st.plotly_chart(fig_patient_line, use_container_width=True)

    st.markdown('---')
    st.markdown('### Claims & Insurance')
    col_claim1, col_claim2 = st.columns([1,2])
    with col_claim1:
        st.markdown('<div style="font-size:22px;color:#fff;">Overall Claims Count</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:48px;font-weight:700;color:#fff;">{int(selected_row["claims_count"]):,}</div>', unsafe_allow_html=True)
    with col_claim2:
        st.markdown('<div style="font-size:22px;color:#fff;">Claims Count by Gender</div>', unsafe_allow_html=True)
        fig_patients_gender = go.Figure(data=[go.Pie(labels=gender_labels, values=claims_gender, hole=.5)])
        fig_patients_gender.update_layout(title_text='', showlegend=True)
        st.plotly_chart(fig_patients_gender, use_container_width=True)

    claim_month_cols = [col for col in df.columns if col.startswith('claim_count_2024-')]
    claim_months = [col.split('_')[-1] for col in claim_month_cols]
    claim_month_values = [int(selected_row[col]) for col in claim_month_cols]
    fig_claim_line = go.Figure()
    fig_claim_line.add_trace(go.Scatter(x=claim_months, y=claim_month_values, mode='lines+markers', name='Claims'))
    fig_claim_line.update_layout(title_text='Claims Count by Months', xaxis_title='Month', yaxis_title='Count')
    st.plotly_chart(fig_claim_line, use_container_width=True)

    payer_types = ['Aetna', 'Blue Cross', 'Cigna', 'UnitedHealthcare']
    payer_counts = [int(selected_row[payer]) for payer in payer_types]
    sorted_payer_pairs = sorted(zip(payer_types, payer_counts), key=lambda x: x[1])
    sorted_payer_types, sorted_payer_counts = zip(*sorted_payer_pairs)
    colors = ["#d2ebfd", "#a3cb5f", "#28a39d", "#28a39d"]
    fig_payer_bar = go.Figure([go.Bar(x=sorted_payer_types, y=sorted_payer_counts, marker_color=colors)])
    fig_payer_bar.update_layout(title_text='Patients by Insurance Payer', xaxis_title='Payer', yaxis_title='Count')
    st.plotly_chart(fig_payer_bar, use_container_width=True)

    payer_types = ['Medicare', 'Medicaid', 'commercial', 'cash']
    payer_counts = [int(selected_row[payer]) for payer in payer_types]
    sorted_payer_pairs = sorted(zip(payer_types, payer_counts), key=lambda x: x[1])
    sorted_payer_types, sorted_payer_counts = zip(*sorted_payer_pairs)
    colors = ["#f8d0d3", "#d2ebfd", "#7fbdeb", "#4589cb"]
    fig_payer_bar = go.Figure([go.Bar(x=sorted_payer_types, y=sorted_payer_counts, marker_color=colors)])
    fig_payer_bar.update_layout(title_text='Patients by Insurance Coverage', xaxis_title='Payer', yaxis_title='Count')
    st.plotly_chart(fig_payer_bar, use_container_width=True)
    

    commercial_types = ['BrandA', 'BrandB', 'BrandC']
    commercial_counts = [int(selected_row[brand]) for brand in commercial_types]
    sorted_pairs = sorted(zip(commercial_types, commercial_counts), key=lambda x: x[1])
    sorted_types, sorted_counts = zip(*sorted_pairs)
    colors = ["#d2ebfd", "#7fbdeb", "#4589cb"]
    fig_commercial_bar = go.Figure([go.Bar(x=sorted_types, y=sorted_counts, marker_color=colors)])
    fig_commercial_bar.update_layout(title_text='Patients by Brand', xaxis_title='Brand', yaxis_title='Count')
    st.plotly_chart(fig_commercial_bar, use_container_width=True)

    st.markdown('---')
    st.markdown('### Prescription')
    col_rx1, col_rx2 = st.columns(2)
    with col_rx1:
        st.metric('NRX (New Prescriptions)', int(selected_row['nrx']))
    with col_rx2:
        st.metric('TRX (Total Prescriptions)', int(selected_row['trx']))

    st.markdown('---')
    st.markdown('### Call Progress')
    planned_calls = selected_row['planned_calls']
    calls_made = selected_row['calls_made']
    remaining_calls = planned_calls - calls_made
    st.progress(calls_made / planned_calls)
    st.markdown(f'<div style="font-size:18px;">{int(calls_made)} out of {planned_calls} Face to Face calls/meets done with HCP <b>{selected_row["hcp_name"]}</b> ({remaining_calls} remaining)</div>', unsafe_allow_html=True)

    progress_ratio = calls_made / planned_calls
    milestones = [0.25, 0.5, 0.75, 1.0]
    milestone_labels = ['25%', '50%', '75%', '100%']
    stepper = ''
    for i, m in enumerate(milestones):
        if progress_ratio >= m:
            color = '#4caf50'
        else:
            color = '#e0e0e0'
        stepper += f'<span style="display:inline-block;width:60px;height:16px;background:{color};border-radius:8px;margin-right:4px;text-align:center;color:#fff;font-size:14px;line-height:16px;">{milestone_labels[i]}</span>'
    st.markdown(f'<div style="margin-top:8px;margin-bottom:8px;">{stepper}</div>', unsafe_allow_html=True)

    if calls_made == planned_calls:
        st.success('🎉 All planned calls completed!')
    elif calls_made >= int(0.75 * planned_calls):
        st.info('Almost there! Just a few more calls to reach your goal.')
    elif calls_made < int(0.25 * planned_calls):
        st.warning('You are just getting started. Try to make more calls!')