import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

tab1, tab2, tab3 = st.tabs(["Overview", "Visualizations", "I/O"])
with tab1:
    st.markdown("Here is our data:")
    df=pd.read_csv('DS_DA_BS.csv')
    #Clean salary column: we only want yearly salaries
    for ind in df.index:
        if re.search("Hour|-1",df['Salary Estimate'][ind]):
            df.drop([ind],inplace=True)

    #Clean null value
    df.replace(['-1'], [np.nan], inplace=True)
    df.replace(['-1.0'], [np.nan], inplace=True)
    df.replace([-1], [np.nan], inplace=True)

    #Find min, max, and mean salaries
    def extract_numbers_max(x):
        temp=re.findall(r'[0-9]+',x)
        return max([int(i) for i in temp])

    def extract_numbers_min(x):
        temp=re.findall(r'[0-9]+',x)
        return min([int(i) for i in temp])

    df['min salary']=df['Salary Estimate'].apply(extract_numbers_min)
    df['max salary']=df['Salary Estimate'].apply(extract_numbers_max)
    df['mean salary']=(df['min salary']+df['max salary'])/2

    #extract states
    def state_extract(x):
        return x.split(',')[-1]
    df['states']=df['Location'].apply(state_extract)

    st.dataframe(data=df[['Job Title','role','states','Sector','Company Name','mean salary']],width=None, height=None)

    st.markdown("DS,DA,BA Salaries Comparison")
    DS_salaries=list(df[df['role']=='DS']['mean salary'])
    DA_salaries=list(df[df['role']=='DA']['mean salary'])
    BA_salaries=list(df[df['role']=='BA']['mean salary'])
    hist_data=[DS_salaries,DA_salaries,BA_salaries]
    fig=ff.create_distplot(
            hist_data, ['DS','DA','BA'])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.groupby(['role'])['mean salary'].aggregate(['count','mean','std','median']).sort_values(by='mean',ascending=False))
    # sns.distplot(df[df['role']=='DS']['mean salary'],label="DS")
    # sns.distplot(df[df['role']=='DA']['mean salary'],label="DA")
    # sns.distplot(df[df['role']=='BA']['mean salary'],label="BA")
    # plt.legend()
with tab2:
    
    selected_states=[' TX',' CA',' NY',' IL',' AZ',' PA',' FL',' OH',' NJ']
    df_states_DS=pd.DataFrame(
        np.zeros((len(selected_states),3)),
        index=selected_states,
        columns=['DS','DA','BA'])
    for i in selected_states:
        for j in ['DS','DA','BA']:
            df_states_DS.at[i,j]=len(
                df[(df['states']==i) & (df['role']==j)]
            )
    ax=df_states_DS.head(8).plot.bar(stacked=True)
    plt.xlabel('State')
    plt.ylabel('Job Count')
    plt.title('Top 8 States for DS/DA/BA')
    # st.pyplot(ax.get_figure())

    st.pyplot()
    color = plt.cm.plasma(np.linspace(0,1,9))
    ax1=df['Sector'].value_counts().sort_values(ascending=False).head(9).plot.bar(color=color)
    plt.title("Sector with highest number of Jobs in DS/DA/BA")
    plt.xlabel("Sector")
    plt.ylabel("Count")
    st.pyplot(ax1.get_figure())

    
with tab3:
    col1, col2= st.columns(2)
    with col1:
        options = st.multiselect(
            'What states you want to compare',
            [' TX',' CA',' NY',' IL',' AZ',' PA',' FL',' OH',' NJ'])
        st.write('You selected:', options)
        df_filtered_states=df[df.states.isin(options)]

        data_states=df_filtered_states.groupby('states')[['min salary','max salary','mean salary']].mean().sort_values(['mean salary','min salary','max salary'],ascending=False)
        fig = go.Figure()

        fig.add_trace(go.Bar(
        x = data_states.index,
        y = data_states['mean salary'],
        name = 'Mean Salary'
        ))

        fig.add_trace(go.Bar(
        x = data_states.index,
        y = data_states['min salary'],
        name = 'Minimum Salary'
        ))

        fig.add_trace(go.Bar(
        x = data_states.index,
        y = data_states['max salary'],
        name = 'Maximum Salary'
        ))

        fig.update_layout(title = 'Salaries in States', barmode = 'group')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        options1 = st.multiselect('Which sectors you want to compare',['Information Technology','Business Services','Finance','Health Care','Biotech & Pharmaceuticals'
                    ,'Insurance','Manufacturing','Education','Government'])
        st.write('You selected:', options1)

        df_filtered_sectors=df[df.Sector.isin(options1)]
        data_sector=df_filtered_sectors.groupby('Sector')[['min salary','max salary','mean salary']].mean().sort_values(['mean salary','min salary','max salary'],ascending=False)
        st.dataframe(data_sector)
        fig = go.Figure()
        fig.add_trace(go.Bar(
        x = data_sector.index,
        y = data_sector['mean salary'],
        name = 'Mean Salary'
        ))
        fig.add_trace(go.Bar(
        x = data_sector.index,
        y = data_sector['min salary'],
        name = 'Minimum Salary'
        ))
        fig.add_trace(go.Bar(
        x = data_sector.index,
        y = data_sector['max salary'],
        name = 'Maximum Salary'
        ))
    
        fig.update_layout(title = 'Salaries in Different Sectors', barmode = 'group')
        st.plotly_chart(fig, use_container_width=True)