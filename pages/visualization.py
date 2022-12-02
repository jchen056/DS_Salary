import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

st.sidebar.markdown("# Visualization")
tab1, tab2,tab4,tab5, tab3 = st.tabs(["Overview", "States","Sectors","Company Size/Revenue", "I/O"])
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
        st.dataframe(data=df[['Job Title','role','states','Sector','Company Name','Size','Revenue','mean salary']],width=None, height=None)

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

with tab5:
    df['Revenue'] = df['Revenue'].replace('Unknown / Non-Applicable', None)
    df['Revenue'] = df['Revenue'].str.replace('$', ' ')
    df['Revenue'] = df['Revenue'].str.replace('(USD)', ' ')
    df['Revenue'] = df['Revenue'].str.replace('(', ' ')
    df['Revenue'] = df['Revenue'].str.replace(')', ' ')
    df['Revenue'] = df['Revenue'].str.replace(' ', '')
    df['Revenue'] = df['Revenue'].str.replace('2to5billion','2billionto5billion')
    df['Revenue'] = df['Revenue'].str.replace('1to2billion','1billionto2billion')
    df['Revenue'] = df['Revenue'].str.replace('5to10billion','5billionto10billion')
    df['Revenue'] = df['Revenue'].replace('10+billion', '10billionto11billion')
    df['Revenue'] = df['Revenue'].str.replace('Lessthan1million', '0millionto1million')
    df['Revenue'] = df['Revenue'].str.replace('million', ' ')
    df['Revenue'] = df['Revenue'].str.replace('billion', '000 ')
    df=df[df['Revenue'].isin(['100to500 ', '500 to1000 ', '10000 to11000 ', '25to50 ',
       '5000 to10000 ', '5to10 ', '50to100 ', '1000 to2000 ',
       '2000 to5000 ', '0 to1 ', '1to5 ', '10to25 '])]
    df=df[df['Size'].isin(['51 to 200 employees', '1001 to 5000 employees',
       '501 to 1000 employees', '10000+ employees', '1 to 50 employees',
       '201 to 500 employees', '5001 to 10000 employees'])]
    
    avg_revenues=['0 to1 ','1to5 ','5to10 ', '10to25 ',
              '25to50 ','50to100 ','100to500 ', '500 to1000 ', 
              '1000 to2000 ','2000 to5000 ',
       '5000 to10000 ', '10000 to11000 ']
    avg_sizes=['1 to 50 employees',
            '51 to 200 employees', '201 to 500 employees',
            '501 to 1000 employees','1001 to 5000 employees',
            '5001 to 10000 employees', '10000+ employees']
    df_size_revenue=pd.DataFrame(np.zeros((len(avg_revenues), len(avg_sizes) )), index=avg_revenues,columns=avg_sizes)
    for i in avg_revenues:
        for j in avg_sizes:
            df_size_revenue.at[i,j]=len(df[
            (df['Revenue']==i) & (df['Size']==j)])
    fig=px.imshow(df_size_revenue,text_auto=True,
    labels=dict(x="Company Size",y="Millions of $",color="Count"),
    aspect="auto")
    st.plotly_chart(fig)
    # fig, ax = plt.subplots()
    # im = ax.imshow(df_size_revenue,interpolation='nearest')

    # ax.set_yticks(np.arange(len(avg_revenues)), labels=avg_revenues)
    # ax.set_xticks(np.arange(len(avg_sizes)), labels=avg_sizes)
    # plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
    #         rotation_mode="anchor")
    
    # plt.title('Company Size and Revenue(M $)')

    # fig.colorbar(im, ax=ax)
    # st.pyplot(fig,clear_figure=True)
    def generate_stack_bar(avg_revenues,col_name):
        df_revenues_DS=pd.DataFrame(
                np.zeros((len(avg_revenues),3)),
                index=avg_revenues,
                columns=['DS','DA','BA'])
        for i in avg_revenues:
            for j in ['DS','DA','BA']:
                df_revenues_DS.at[i,j]=len(
                    df[(df[col_name]==i) & (df['role']==j)]
                )

        ax=df_revenues_DS.plot.bar(stacked=True)
        plt.ylabel('Job Count')
        fig1=ax.get_figure()
        st.pyplot(fig1,clear_figure=True)
    
    

    def generate_bar_charts(avg_revenues,col_name):
        df_revenues_DS=pd.DataFrame(
                np.zeros((len(avg_revenues),3)),
                index=avg_revenues,
                columns=['min salary','max salary','mean salary'])
        for i in avg_revenues:
            df_revenues_DS.at[i]=df[df[col_name]==i][['min salary','max salary','mean salary']].mean()
        # st.dataframe(df_revenues_DS)
        data_sector=df_revenues_DS
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
        st.plotly_chart(fig, use_container_width=True)
    def generate_df_salries(avg_revenues,col_name):
        df_revenues_DS=pd.DataFrame(
                np.zeros((len(avg_revenues),3)),
                index=avg_revenues,
                columns=['min salary','max salary','mean salary'])
        for i in avg_revenues:
            df_revenues_DS.at[i]=df[df[col_name]==i][['min salary','max salary','mean salary']].mean()
        st.dataframe(df_revenues_DS)

    c_rev,c_size=st.columns(2)
    with c_rev:
        st.markdown("Job Count by Company Revenue")
        generate_stack_bar(avg_revenues,"Revenue")
        generate_df_salries(avg_revenues,"Revenue")
    with c_size:
        st.markdown("Job Count by Company Size")
        generate_stack_bar(avg_sizes,"Size")
        generate_df_salries(avg_sizes,"Size")
    st.markdown("---")
    st.markdown("**Salaries by Company Revenue**")
    generate_bar_charts(avg_revenues,"Revenue")
    st.markdown("**Salaries by Company Size**")
    generate_bar_charts(avg_sizes,"Size")
    st.markdown("---")
    st.markdown('''
    * The higher the revenue, the bigger the company size.
    * More jobs can be found in really big corporations(10 billion+ in revenue and 10000+ employees).
    * Salaries about the same across companies of different revenues and sizes. Really big corporations(10 billion+ in revenue and 10000+ employees) offer more salaries.'''
    )
    

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