print("adagrad")
#Import untilities
print("-----------------------------")
print("Importing libraries")
import pandas as pd
import numpy as np
import teradata
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#Import Synthetic Minority Over-Sampling Technique algorithm
from imblearn.over_sampling import SMOTE

#Import Neural Network Libraries
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Dense
#from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras import optimizers
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
import numpy as np
import time
from sklearn import metrics
print("Libraries imported")
print("-----------------------------\n")

#Import and massage data
print("-----------------------------")
print("Importing dataset")
df = pd.read_pickle('df_3')
print("Dataset Imported")
print("-----------------------------\n")


##Setting up Teradata utils
#def generate_dataframe_from_teradata(username, password, connection, sql, appname='HelloWorld'):
#    a=datetime.datetime.now()
#    print ('Initiating the conection module.')
#    udaExec = teradata.UdaExec (appName=appname, version="1.0",
#                                logConsole=False)
#    print ('Connecting to the database %s as %s'%(connection,username))
#    try:
#        session = udaExec.connect(method="odbc", system=connection,
#                              username=username, password=password, MechanismName="LDAP")
#        print ('Connection Established. Executing the query')
#        df = pd.read_sql_query(sql,session)
#        print ('Done.')
#        return df
#    except:
#        print ("Something Went Wrong. Please check your SQL and Authentication Information.")

#def make_df(username,password,connection,sql, appname='HelloWorld'):
#    a=datetime.datetime.now()
#    udaExec = teradata.UdaExec (appName, version="1.0",
#                               logConsole=False)
#    print ('Connecting to the database %s as %s')


#Query

#sql = """select distinct 
#Inquiry,
#DT_ID,

#case when CASH_OUT_FLG = 1 then 1 else 0 end CASH_OUT_FLAG,
#case when CASH_IN_FLG = 1 then 1 else 0 end CASH_IN_FLAG,
#case when USAA_FLG = 1 then 1 else 0 end USAA_FLAG,
#case when VETERAN_FLG = 1 then 1 else 0 end VET_FLAG,
#case when COUNTRY_DESC = 'UNITED STATES' then 1 else 0 end US_FLAG,
#case when REPEATINQUIRIES = 1 then 1 else 0 end REPEAT_INQUIRIES,
#
#case when USABLEDISPLAYWIDTH = 'N/A' then 0 else USABLEDISPLAYWIDTH end USABLEDISPLAYWIDTH,
#case when USABLEDISPLAYHEIGHT = 'N/A' then 0 else USABLEDISPLAYHEIGHT end USABLEDISPLAYHEIGHT,
#case when DISPLAYPPI = 'N/A' then 0 else DISPLAYPPI end DISPLAYPPI,
#case when DIAGONALSCREENSIZE = 'N/A' then 0 else DIAGONALSCREENSIZE end DIAGONALSCREENSIZE,
#
#case when CREATIVE_CATEGORY_DESC = 'MSQL' then 'MSQL'
#          when CREATIVE_CATEGORY_DESC = 'MSQL-LEND' then 'MSQL-LEND'
#          else 'LRE' end as CREATIVE_CATEGORY_DESC,
#  
#case when PHONE_TYPE_DESC = 'Land line, non-wireless service including POTS, Broadband, etc.' then 'Land_line'
#          when PHONE_TYPE_DESC = 'Wireless type service including PCS, Cellular, GSM, etc.' then 'Wireless'
#          else 'Other' end as PHONE_TYPE_DESC,
#
#case when OSNAME like '%Windows%' then 'Windows' when OSNAME = 'Android' then 'Android'
#          when OSNAME in ('iOS, OS X') then 'Apple'  else 'Other' end  OS_NAME,
#  
#case when BUSINESS_NAME like '%AT&T%' then 'AT&T' when BUSINESS_NAME like '%Verizon%' then 'Verizon' when BUSINESS_NAME like '%Sprint%' then 'Sprint'
#          when BUSINESS_NAME like '%Metro%' then 'Metro_PCS' else 'Other' end P_CARRIER,
#  
#case when SITE_NAME like '%instagram%' then 'instagram' when SITE_NAME like '%facebook%' then 'FB' when SITE_NAME like '%google%' then 'google' when SITE_NAME like '%yahoo%' then 'yahoo'
#          when SITE_NAME like '%taboola%' then 'taboola' when SITE_NAME like '%aol%' then 'aol' when SITE_NAME like '%msn%' then 'msn' when SITE_NAME like '%pinterest%' then 'pinterest' else 'other' end SITE_NAME,
#
#case when MANUFACTURER = 'ZTE' then 'ZTE' when MANUFACTURER = 'Apple' then 'Apple' when MANUFACTURER = 'HTC' then 'HTC_google' when MANUFACTURER = 'Google' then 'HTC_google' 
#          when MANUFACTURER = 'Motorola' then 'Moto' when MANUFACTURER = 'Microsoft' then 'Microsoft' when MANUFACTURER = 'Samsung' then 'Samsung' when MANUFACTURER = 'LG' then 'LG' else 'Other' end PHONE_MAKE,
#
#case when BROWSERNAME = 'Chrome' then 'Chrome' when BROWSERNAME = 'Chrome Mobile' then 'Chrome_Mobile' when BROWSERNAME = 'Edge' then 'Edge' when BROWSERNAME = 'Safari' then 'Safari' 
#          when BROWSERNAME = 'Internet Explorer' then 'int_explorer' when BROWSERNAME = 'Internet Explorer Mobile' then 'int_explorer_mobile' when BROWSERNAME = 'Firefox' then 'firefox'
#  when BROWSERNAME = 'Samsung Browser' then 'samsung_browser' else 'Other' end BROWSER,
#
#case when CHANNEL_TYPE_DESC = 'searchengine' then 'search' when CHANNEL_TYPE_DESC = 'native' then 'native' when CHANNEL_TYPE_DESC = 'social' then 'social' when CHANNEL_TYPE_DESC = 'media' then 'media'
#          when CHANNEL_TYPE_DESC = 'affiliate' then 'affiliate' else 'other' end CHANNEL,
#
#case when SERVICE_RATE = 1 then '1' when SERVICE_RATE = '2' then 2 else '3+' end SERVICE_RATE,
#
#LOAN_AMT_NEEDED,
#DAY_PHONE_MSA_POPULATION,
#CREDIT_PROFILE_PARENT_DESC,
#PROPERTY_MORTGAGE_1_INT,
#AGE,
#OCCUPATION_STATUS_PARENT_DESC,
#STATE,
#PRIMARYHARDWARETYPE,
#SESSION_TIMESTAMP,
#COMPLETE_STATUS_DTTM,
#LAND_AREA,
#CITY_DESC,
#DAY_OF_WEEK_ID,
#HOUR_ID1,
#FIRST_NAME,
#HOME_VALUE,
#LOAN_TO_VALUE_RATIO,
#
#case when APPLICATIONS = 1 then 1 else 0 end APPLICATIONS
#
#from dstoltz.QL_Quality_Jan_Apr2017"""

#Run query and generate dataset
#username,password='ggiorcelli','BigData1'
#df=generate_dataframe_from_teradata(username,password,'TDPRD',sql)

#Pickle the dataset
#df.to_pickle('df_1')
#pd.set_option('display.max_columns', None)
#df.head(5)

#Add gender
names = pd.read_csv('name_gender_lookup.csv')

names['first_name'] = names['first_name'].str.lower()
df['FIRST_NAME'] = df['FIRST_NAME'].str.lower()
names.columns = ['FIRST_NAME','gender']

df = pd.merge(df, names, on='FIRST_NAME', how='left')
df['m_flag'] = np.where(df.gender == 'M', 1, 0)

#Create session time
df['SESSION_TIMESTAMP'] = pd.to_datetime(df['SESSION_TIMESTAMP'])
df['COMPLETE_STATUS_DTTM'] = pd.to_datetime(df['COMPLETE_STATUS_DTTM'])

df['SESSION_TIME'] = (df.COMPLETE_STATUS_DTTM - df.SESSION_TIMESTAMP).astype('timedelta64[s]')

#Map states to region
states = pd.read_csv('STATE_REGION.csv')
df = pd.merge(df, states, on='STATE', how='left')

df['REGION'] = df['REGION'].fillna('OTHER')

#Map cities to city sizes
cities = pd.read_csv('CITY_SIZE.csv')

df['CITY_DESC'] = df['CITY_DESC'].str.lower()
cities['CITY_DESC'] = cities['CITY_DESC'].str.lower()

df = pd.merge(df, cities, on='CITY_DESC', how='left')
df['CITY_SIZE'] = df['CITY_SIZE'].fillna('S')


###Some data munging
#Credit encoding
credit_encoding = { "CREDIT_PROFILE_PARENT_DESC": {"Excellent": 5, "Good": 4, "Needs Improvement": 3,
                                                   "Establishing Credit": 1, "N/A": 2}}
df.replace(credit_encoding, inplace=True)

#Hardware type
def f(row):
    if row['PRIMARYHARDWARETYPE'] == 'Tablet':
        val = 'Tablet'
    elif row['PRIMARYHARDWARETYPE'] == 'Mobile Phone':
        val = 'Mobile_Phone'
    elif row['PRIMARYHARDWARETYPE'] == 'Desktop':
        val = 'Desktop'     
    else:
        val = 'other'
    return val

df['PRIMARYHARDWARETYPE'] = df.apply(f, axis=1)

###Binning
bins = [-1, 1, 500, 800, 1200, 1700]
group_names = ['0','1', '2', '3', '4']
df.USABLEDISPLAYWIDTH = df.USABLEDISPLAYWIDTH.astype(int)
df['USABLEDISPLAYWIDTH'] = pd.cut(df['USABLEDISPLAYWIDTH'], bins, labels=group_names)

df.USABLEDISPLAYHEIGHT = df.USABLEDISPLAYHEIGHT.astype(int)
bins = [-2, 1, 500, 600, 700, 800, 1200, 1700, 2200]
group_names = ['0','1', '2', '3', '4','5','6','7']
df['USABLEDISPLAYHEIGHT'] = pd.cut(df['USABLEDISPLAYHEIGHT'], bins, labels=group_names)

df.DIAGONALSCREENSIZE = df.DIAGONALSCREENSIZE.astype(float)
df.DISPLAYPPI = df.DISPLAYPPI.astype(float)
bins = [-2, 100, 220, 320, 400, 475, 575, 700, 2000]
group_names = ['0','1', '2', '3', '4', '5', '6', '7']
df['DISPLAYPPI'] = pd.cut(df['DISPLAYPPI'], bins, labels=group_names)

bins = [-1,120, 180, 240, 300, 360, 420, 500, 1000, 2000, 20000]
group_names = ['0','1', '2', '3', '4', '5', '6', '7','8','9']
df['SESSION_TIME'] = pd.cut(df['SESSION_TIME'], bins, labels=group_names)

bins = [-1,3,6,9,12,15,18,21,25]
group_names = ['0','1', '2', '3', '4', '5', '6', '7']
df['HOUR_ID1'] = pd.cut(df['HOUR_ID1'], bins, labels=group_names)

###One hot encoding for categorical variables
df = pd.get_dummies(data=df, columns=['CREATIVE_CATEGORY_DESC','PHONE_TYPE_DESC', 'OS_NAME', 'P_CARRIER',
                                 'SITE_NAME', 'PHONE_MAKE', 'BROWSER','CHANNEL','SERVICE_RATE', 
                                 'OCCUPATION_STATUS_PARENT_DESC', 'PRIMARYHARDWARETYPE', 'DAY_OF_WEEK_ID',
                                 'HOUR_ID1','SESSION_TIME','REGION','CITY_SIZE'])

cols = list(df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('APPLICATIONS'))
df = df[cols+['APPLICATIONS']]

###Cleaning up the dataset
#Getting rid of all variables that have been used already
df.drop('FIRST_NAME', axis=1, inplace=True)
df.drop('gender', axis=1, inplace=True)
df.drop('SESSION_TIMESTAMP', axis=1, inplace=True)
df.drop('COMPLETE_STATUS_DTTM', axis=1, inplace=True)
df.drop('CITY_DESC', axis=1, inplace=True)
df.drop('STATE', axis=1, inplace=True)
df.drop('Inquiry', axis=1, inplace=True)
df.drop('DT_ID', axis=1, inplace=True)


###Preparing the last fields for modeling
#Filling NAs with mean and normalizing all contnuous variables
df.USABLEDISPLAYWIDTH = df.USABLEDISPLAYWIDTH.astype(float)
df.USABLEDISPLAYHEIGHT = df.USABLEDISPLAYHEIGHT.astype(float)
df.DISPLAYPPI = df.DISPLAYPPI.astype(float)

df['USABLEDISPLAYWIDTH'] = df['USABLEDISPLAYWIDTH'].fillna(df['USABLEDISPLAYWIDTH'].mean())
df['DAY_PHONE_MSA_POPULATION'] = df['DAY_PHONE_MSA_POPULATION'].fillna(df['DAY_PHONE_MSA_POPULATION'].mean())
df['LAND_AREA'] = df['LAND_AREA'].fillna(df['LAND_AREA'].mean())

df['LOAN_AMT_NEEDED'] = np.log(df['LOAN_AMT_NEEDED'].loc[df['LOAN_AMT_NEEDED'] != 0])
df['DAY_PHONE_MSA_POPULATION'] = np.log(df['LOAN_AMT_NEEDED'].loc[df['LOAN_AMT_NEEDED'] != 0])
df['LAND_AREA'] = np.log(df['LAND_AREA'].loc[df['LAND_AREA'] != 0])
df['HOME_VALUE'] = np.log(df['HOME_VALUE'].loc[df['HOME_VALUE'] != 0])
df['AGE'] = np.log(df['AGE'].loc[df['AGE'] != 0])
df['PROPERTY_MORTGAGE_1_INT'] = np.log(df['PROPERTY_MORTGAGE_1_INT'].loc[df['PROPERTY_MORTGAGE_1_INT'] != 0])
df['DISPLAYPPI'] = np.log(df['DISPLAYPPI'].loc[df['DISPLAYPPI'] != 0])


#Splitting train, test and validation
training_features, test_features, training_target, test_target, = train_test_split(df.drop(['APPLICATIONS'], axis=1),
                                               df['APPLICATIONS'],
                                               test_size = .015,
                                               random_state=12)

x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = 0.15,
                                                  random_state=12)
												  
#Last clean-up before syntetic resampling
df['LOAN_AMT_NEEDED'] = df['LOAN_AMT_NEEDED'].fillna(df['LOAN_AMT_NEEDED'].mean())
df['DAY_PHONE_MSA_POPULATION'] = df['DAY_PHONE_MSA_POPULATION'].fillna(df['DAY_PHONE_MSA_POPULATION'].mean())
df['HOME_VALUE'] = df['HOME_VALUE'].fillna(df['HOME_VALUE'].mean())
df['PROPERTY_MORTGAGE_1_INT'] = df['PROPERTY_MORTGAGE_1_INT'].fillna(df['PROPERTY_MORTGAGE_1_INT'].mean())


from imblearn.over_sampling import SMOTE

#Using Synthetic Minority Over-Sampling techinque to balance the classes
print("-----------------------------")
print("Applying SMOTE to train and val subset")
sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
x_val, y_val = sm.fit_sample(x_val, y_val)
print("Over-sampling completed")
print("-----------------------------\n")

#Check SMOTE results
print("Below are the results from the SMOTE algorithm")
print("Y=1 in raw train subset:", y_train.sum()) 
print("Y=1 in resampled train subset:", y_train_res.sum())
print("\n")


#Building the neural network
model = Sequential()
model.add(Dense(400, activation = 'relu', kernel_initializer='normal', input_dim = x_train_res.shape[1]))   #400 for max ROC
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
adagrad = Adagrad(lr=0.003, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer = adagrad, metrics = ['accuracy'])


#Fit neural network
print("-----------------------------")
print("Training neural network")
y_train_res = y_train_res.reshape((-1, 1))
x_train_res, y_train_res = shuffle(x_train_res, y_train_res)

#x_val = x_val.as_matrix()
#y_val = y_val.as_matrix()

start = time.time()
model_hist = model.fit(x_train_res, y_train_res, batch_size = 32, epochs = 50, validation_data=(x_val, y_val))
end = time.time()
print('Train time:', end - start, 'sec')
print("-----------------------------\n")

#Save neural network
#print("-----------------------------")
#print("Saving neural network")
#model_json = model.to_json()   #serialize model to json
#with open("model.json", "w") as json_file:
    #json_file.write(model_json)

#model.save_weights("model.h5") #serialize weights to HDF5
#print("Saved model to disk")
#print("-----------------------------\n")


#Use the code below to load the model and its weights
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()   #Load json and create model
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)

#loaded_model.load_weights("model.h5")  #Load weights into new model
#print("Loaded model from disk")


#List all data in history
print(model_hist.history.keys())

#Summarize history for accuracy
plt.plot(model_hist.history['acc'])
plt.plot(model_hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Summarize history for loss
plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Calculate accuracy and ROC on test data
test_features = test_features.as_matrix()
test_target = test_target.as_matrix()

scores = model.evaluate(test_features, test_target, verbose=1)
print("\n-----------------------------")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("Recall score", recall_score(test_target, np.where(model.predict(test_features) > 0.5, 1, 0)))
print("-----------------------------\n")


phat = model.predict_proba(test_features)#[:,1]
print("\n-----------------------------")
print("ROC Score: ",metrics.roc_auc_score(test_target, phat))
print("-----------------------------\n")


#Build ROC score and curve
phat = model.predict_proba(test_features)#[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_target, phat)
roc_auc = metrics.auc(fpr, tpr)


#Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

###End