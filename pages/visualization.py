import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

st.sidebar.markdown("# Visualization")
tab1, tab2,tab4, tab3 = st.tabs(["Overview", "States","Sectors", "I/O"])
with tab1:
    
    df=pd.read_csv('pages/DS_DA_BS.csv')
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
    with st.expander("Click here to view the data"):
        st.markdown("Here is our data:")
        st.dataframe(data=df[['Job Title','role','states','Sector','Company Name','mean salary']],width=None, height=None)

    st.markdown("**DS,DA,BA Salaries Comparison**")
    DS_salaries=list(df[df['role']=='DS']['mean salary'])
    DA_salaries=list(df[df['role']=='DA']['mean salary'])
    BA_salaries=list(df[df['role']=='BA']['mean salary'])
    hist_data=[DS_salaries,DA_salaries,BA_salaries]
    fig=ff.create_distplot(
            hist_data, ['DS','DA','BA'])
    st.plotly_chart(fig, use_container_width=True)
    col1, col2=st.columns(2)
    with col1:
        st.dataframe(df.groupby(['role'])['mean salary'].aggregate(['count','mean','std','median']).sort_values(by='mean',ascending=False))
        st.caption('The table above shows the job count for DS, BA, and DA. In addition, mean, std, and median of salaries are displayed in the table.')
    with col2:
        st.markdown('''
* A **Data Scientist** specializes in high-level data manipulation, including writing complex algorithms and computer programming. **Business Analysts** are more focused on creating and interpreting reports on how the business is operating day to day, and providing recommendations based on their findings.
* Data analysts and business analysts both help drive data-driven decision-making in their organizations. Data analysts tend to work more closely with the data itself, while business analysts tend to be more involved in addressing business needs and recommending solutions.''')
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
    fig1=ax.get_figure()
    st.pyplot(fig1,clear_figure=True)

    df_ny=df[df['states']==' NY']
    df_tx=df[df['states']==' TX']
    df_ca=df[df['states']==' CA']
    p1,p2,p3=st.columns(3)

    def plotting_function(df):
        DS_salaries=list(df[df['role']=='DS']['mean salary'])
        DA_salaries=list(df[df['role']=='DA']['mean salary'])
        BA_salaries=list(df[df['role']=='BA']['mean salary'])
        
        hist_data=[DS_salaries,DA_salaries,BA_salaries]
        fig=ff.create_distplot(
                hist_data, ['DS','DA','BA'])
        st.plotly_chart(fig,use_container_width=True)
    with p1:
        st.markdown("### New York")
        # plotting_function(df_ny)
        df_ny_info=df_ny.groupby(['role'])['mean salary'].aggregate(['count','mean','std','median']).sort_values(by='mean',ascending=False)
        st.dataframe(df_ny_info,use_container_width=True)
    with p2:
        st.markdown("### Texas")
        # plotting_function(df_tx)
        st.dataframe(df_tx.groupby(['role'])['mean salary'].aggregate(['count','mean','std','median']).sort_values(by='mean',ascending=False),use_container_width=True)
    with p3:
        st.markdown("### California")
        # plotting_function(df_ca)
        st.dataframe(df_ca.groupby(['role'])['mean salary'].aggregate(['count','mean','std','median']).sort_values(by='mean',ascending=False),use_container_width=True)
    
    st.markdown('''
    * More jobs found in TX and CA
    * NY and CA offer higher salaries''')
    


with tab4:
    # st.markdown("states")
    # st.pyplot()
    color = plt.cm.plasma(np.linspace(0,1,9))
    ax1=df['Sector'].value_counts().sort_values(ascending=False).head(9).plot.bar(color=color)
    plt.title("Sector with highest number of Jobs in DS/DA/BA")
    plt.xlabel("Sector")
    plt.ylabel("Count")
    st.pyplot(ax1.get_figure(),clear_figure=True)

    data_sector=df[df['Sector'].isin(['Information Technology','Business Services','Finance','Health Care','Biotech & Pharmaceuticals'
                 ,'Insurance','Manufacturing','Education','Government'])].groupby('Sector')[['min salary','max salary','mean salary']].mean().sort_values(['mean salary','min salary','max salary'],ascending=False).head(8)
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
    st.markdown('''
    * More jobs can be found in IT and Financial Industry
    * Biotech and IT have higher pay''')
with tab3:
    with st.expander("Click to see jobs by states"):
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
        st.success("Graphs generated successfully")

    with st.expander("Click to see jobs by sectors"):
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
        st.success("Graphs generated successfully")