import streamlit as st
import numpy as np
from ranzi_2014 import*
import plotly.express as px

"""
# Biomass Pyrolysis
Estimation of the product formation from fast pyrolysis of lignocellulosic biomass using the kinetic model from [Ranzi et al. (2014)](https://doi.org/10.1016/j.ces.2013.08.014).

The kinetic model is a fork of [ccpcode/kinetic-scheme](https://github.com/ccpcode/kinetic-schemes)
"""
st.sidebar.markdown('# Inputs')
st.sidebar.write('**Composition of biomass in weight%**')
try:
    wtc=float(st.sidebar.text_input('Cellulose', value=50))
    wth=float(st.sidebar.text_input('Hemicellulose', value=15))
    wtl=float(st.sidebar.text_input('Lignin', value=35))
except:
    st.sidebar.markdown(':octagonal_sign: Enter a number')

# normalize the inputs
wtcell = wtc/(wtc+wth+wtl)*100
wthemi = wth/(wtc+wth+wtl)*100
wtlig = wtl/(wtc+wth+wtl)*100

st.sidebar.write('**Temperature**')
temp = st.sidebar.slider('Temperature (K)', min_value=600, max_value=973, value=650, step=1)

st.sidebar.write('**Time**')
try:
    t_max=float(st.sidebar.text_input('Time (s)', value=4))
except:
    st.sidebar.markdown(':octagonal_sign: Enter a number')

# run the kinetic model
df_cell, df_hemi, df_ligc, df_ligh, df_ligo, df_gasprod, df_tarprod, df_total = run_ranzi_2014(wtcell, wthemi, wtlig, temp, t_max)

st.subheader('Main products')
# Main product char
fig1=px.line(df_total, template='plotly')
fig1.update_yaxes(title='Weight fraction (dry basis)', ticks='outside')
fig1.update_xaxes(ticks='outside')
st.plotly_chart(fig1, use_container_width=True)

st.subheader('Gas')
# Main product char
fig2=px.line(df_gasprod, template='plotly')
fig2.update_yaxes(title='Weight fraction (dry basis)', ticks='outside')
fig2.update_xaxes(ticks='outside')
st.plotly_chart(fig2, use_container_width=True)

st.subheader('Tar')
# Main product char
fig3=px.line(df_tarprod, template='plotly')
fig3.update_yaxes(title='Weight fraction (dry basis)', ticks='outside')
fig3.update_xaxes(ticks='outside')
st.plotly_chart(fig3, use_container_width=True)