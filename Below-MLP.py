###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('Below_model_mlp.pkl')
scaler = joblib.load('Below_scaler.pkl') 

# Streamlit user interface
st.title("Sirolimus Low Exposure (<10 ng/mL) Predictor")

# Define feature names
feature_names = ['BMI', 'TG','ABCB1_rs1128503_AA', 'ABCB1_rs1128503_AG',
       'ABCB1_rs1128503_GG', 'mTOR_rs2076655_GG', 'mTOR_rs2076655_AG',
       'mTOR_rs2076655_AA', 'IL10_rs1800896_CT']

BMI = st.number_input("BMI (kg/m2):", min_value=10, max_value=30, value=15)
TG = st.number_input("IL-10.rs1800896 (1=TT,2=CT):", min_value=1, max_value=2, value=1)
IL10_rs1800896_CT = st.number_input("TG (mmol/L):", min_value=0, max_value=6, value=3)
ABCB1_rs1128503_AA = st.selectbox("ABCB1.rs1128503_AA:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
ABCB1_rs1128503_AG = st.selectbox("ABCB1.rs1128503_AG:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
ABCB1_rs1128503_GG = st.selectbox("ABCB1.rs1128503_GG:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
mTOR_rs2076655_GG = st.selectbox("mTOR.rs2076655_GG:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
mTOR_rs2076655_AG = st.selectbox("mTOR.rs2076655_AG:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
mTOR_rs2076655_AA = st.selectbox("mTOR.rs2076655_AA:", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
#IL10_rs1800896_CT = st.selectbox("IL-10.rs1800896_CT:", options=[1, 2], format_func=lambda x: 'No' if x == 1 else 'Yes')


# 准备输入特征
feature_values = [BMI, TG,ABCB1_rs1128503_AA, ABCB1_rs1128503_AG,
       ABCB1_rs1128503_GG, mTOR_rs2076655_GG, mTOR_rs2076655_AG,
       mTOR_rs2076655_AA, IL10_rs1800896_CT]
features = np.array([feature_values])

# 分离连续变量和分类变量
continuous_features = [BMI, TG]
categorical_features=[ABCB1_rs1128503_AA, ABCB1_rs1128503_AG,
       ABCB1_rs1128503_GG, mTOR_rs2076655_GG, mTOR_rs2076655_AG,
       mTOR_rs2076655_AA, IL10_rs1800896_CT]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)

# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array, columns=['BMI', 'TG'])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
categorical_features_array = np.array(categorical_features).reshape(1, -1)

# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized, categorical_features_array])

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=feature_names)


if st.button("Predict"): 
    OPTIMAL_THRESHOLD = 0.532
    
    # Predict class and probabilities    
    #predicted_class = model.predict(final_features_df)[0]   
    predicted_proba = model.predict_proba(final_features_df)[0]
    prob_class1 = predicted_proba[1]  # 类别1的概率

    # 根据最优阈值判断类别
    predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0

    # 显示结果（概率形式更直观）
    st.write(f"**Low Exposure Probability:** {prob_class1:.1%}")
    st.write(f"**Decision Threshold:** {OPTIMAL_THRESHOLD:.0%} (optimized for clinical utility)")
    st.write(f"**Predicted Class:** {predicted_class} (1: High risk, 0: Low risk)")

    # Display prediction results    
    #st.write(f"**Predicted Class:** {predicted_class} (1: High risk, 0: Low risk)")   
    #formatted_proba = ", ".join(f"{prob:.2f}" for prob in predicted_proba)
    #st.write(f"**Prediction Probabilities:** {formatted_proba}")

    # Generate advice based on prediction results  
    #probability = predicted_proba[predicted_class] * 100
    #if predicted_class == 1:        
         #advice = (            
                #f"According to our model, you have a high risk of breast cancer recurrence. "            
                #f"The model predicts that your probability of having breast cancer recurrence is {probability:.1f}%. "                   
          #)    
    #else:        
         #advice = (           
                #f"According to our model, you have a low risk of breast cancer recurrence. "            
                #f"The model predicts that your probability of not having breast cancer recurrence is {probability:.1f}%. "            
          #)    
    #st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")

    # 创建SHAP解释器
    # 假设 X_train 是用于训练模型的特征数据
    df=pd.read_csv('Below_train_lasso.csv',encoding='utf8')
    ytrain=df.Below
    x_train=df.drop('Below',axis=1)
    from sklearn.preprocessing import StandardScaler
    continuous_cols = [1,2]
    xtrain = x_train.copy()
    scaler = StandardScaler()
    xtrain.iloc[:, continuous_cols] = scaler.fit_transform(x_train.iloc[:, continuous_cols])

    explainer_shap = shap.KernelExplainer(model.predict_proba, xtrain)
    
    # 获取SHAP值
    shap_values = explainer_shap.shap_values(pd.DataFrame(final_features_df,columns=feature_names))
    
  # 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)

# Display the SHAP force plot for the predicted class    
    if predicted_class == 1:        
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], original_feature_values, matplotlib=True)    
    else:        
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], original_feature_values, matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)    
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
