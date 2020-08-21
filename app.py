from flask import Flask
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
import re
import os
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
import json
import collections
import pprint

app = Flask(__name__)

cred = credentials.Certificate("./sdk-firebase.json")
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://tanahairku-experimental-cl.firebaseio.com/'
})

ref = firebase_admin.db.reference("/users")

base_category = ["rumah","pakaian", "makanan", "musik", "senjata"]

base_item = {
        "rumah" : [
            "Rumah_Aceh",
            "Rumah_Asmat",
            "Rumah_Sunda",
            "Rumah_Bali",
            "Rumah_Dayak",
            "Rumah_Toraja"
        ],
        "pakaian" : [
            'Pakaian_Aceh',
            'Pakaian_Asmat',
            'Pakaian_Bali',
            'Pakaian_Dayak',
            'Pakaian_Sunda',
            'Pakaian_Toraja'
        ],
        "makanan" : [                
            'Makanan_Aceh',
            'Makanan_Asmat',
            'Makanan_Bali',
            'Makanan_Dayak',
            'Makanan_Sunda',
            'Makanan_Toraja'
        ],
        "musik" : [
            'Musik_Aceh',
            'Musik_Asmat',
            'Musik_Bali',
            'Musik_Dayak',
            'Musik_Sunda',
            'Musik_Toraja'
        ],
        "senjata" : [
            'Senjata_Aceh',
            'Senjata_Asmat',
            'Senjata_Bali',
            'Senjata_Dayak',
            'Senjata_Sunda',
            'Senjata_Toraja'
        ]
}

@app.route('/')
def index():
    return "Hello world"

@app.route("/get-data-frame")
def get_data_frame_handler():
    return getDataFrame().to_json()

@app.route("/get-itemset/<min_sp>")
def get_itemset_handler(min_sp):
    return getItemset(float(min_sp)).to_json()

@app.route("/get-rules/<min_sp>/<min_conf>")
def get_rules_handler(min_sp, min_conf):
    return getRules(float(min_sp),float(min_conf)).to_json()

@app.route("/get-rules-sorted/<min_sp>/<min_conf>")
def get_rules_sorted_handler(min_sp, min_conf):
    return getRulesSorted(float(min_sp),float(min_conf)).to_json()

@app.route("/get-frequency/<min_sp>/<min_conf>")
def get_frequency_handler(min_sp, min_conf):
    itemset = getRulesSorted(float(min_sp),float(min_conf))
    return json.dumps(getFrequencyOfCategory(itemset))

@app.route("/get-suggestion/<min_sp>/<min_conf>")
def get_suggestion_handler(min_sp, min_conf):
    return getSuggestion(min_sp,min_conf).to_json()

@app.route("/get-not-accessed/<min_sp>/<min_conf>")
def get_not_accessed_handler(min_sp, min_conf):
    return (getNotAccessedContent(min_sp, min_conf))

@app.route("/get-final-suggestion/<min_sp>/<min_conf>")
def get_final_suggestion_handler(min_sp, min_conf):
    return getFinalSuggestion(min_sp,min_conf)

@app.route("/get-dirty-data")
def get_dirty_data_handler():
    return getDirtyData().to_json()

def getDirtyData():
    data = ref.get()
    log_data = []
    for userid in data:
        for id_log in data[userid]['log'] :
            log_data.append([
                data[userid]['log'][id_log]['Date'],
                id_log,
                data[userid]['log'][id_log]['Items_Interval'],
                data[userid]['log'][id_log]['Items_Log'],
            ])

    df = pd.DataFrame(log_data)
    return df

def getDataFrame():
    data = ref.get()
    log_data = []
    for userid in data:
        for id_log in data[userid]['log'] :
            log_data.append([
                data[userid]['log'][id_log]['Date'],
                id_log,
                data[userid]['log'][id_log]['Items_Interval'],
                data[userid]['log'][id_log]['Items_Log'],
            ])

    df = pd.DataFrame(log_data)
    new_names = {0 : 'tgl_akses',
             1 : 'id_user',
             2 : 'durasiAkses',
             3 : 'konten'
            }

    df.rename(columns=new_names, inplace=True)
    df.drop(df[df.durasiAkses == '0'].index, inplace=True)
    df.dropna()

    new_konten = [konten.replace('START, ', '') for konten in df.konten]
    dataset = []
    for x in new_konten:
        dataset.append(x.split(", "))

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df

def getItemset(min_sp):
    df = getDataFrame()
    frequent_itemsets_fp = fpgrowth(df, min_support=min_sp, use_colnames=True)
    frequent_itemsets_fp['length'] = frequent_itemsets_fp['itemsets'].apply(lambda x:len(x))
    itemsetFP = frequent_itemsets_fp
    return itemsetFP.sort_values(by='support',ascending=False)
    
def getRules(min_sp, min_conf):
    itemsetFP = getItemset(min_sp)
    rules_fp = association_rules(itemsetFP, metric="confidence", min_threshold=min_conf)
    return rules_fp

def getRulesSorted(min_sp, min_conf, asc=False, sort_metric='lift'):
    rules_fp = getRules(min_sp, min_conf)
    a = rules_fp[['antecedents','consequents','support','confidence','lift']]
    b = a.sort_values(by=sort_metric, ascending=asc)

    return b.reset_index(drop=True)

def getFrequencyOfCategory(itemset):
    base_category_counter = {
        "rumah" : 0,
        "pakaian" : 0,
        "makanan" : 0,
        "musik" : 0,
        "senjata" : 0
    }

    fs = itemset['antecedents']
    for parent in fs:
        for item in parent:
            key = item.split("_")[0].lower()
            base_category_counter[key] += 1
    
    fs = itemset['consequents']
    for parent in fs:
        for item in parent:
            key = item.split("_")[0].lower()
            base_category_counter[key] += 1

    return base_category_counter

def getSuggestion(min_sp, min_conf):
    df = getRulesSorted(float(min_sp), float(min_conf), True, 'confidence')
    tf = pd.DataFrame(columns=df.columns.values.tolist())
    ff = pd.DataFrame(columns=df.columns.values.tolist())

    for index, row in df.iterrows():
        antecedents_flag = {
            "rumah"     : False,
            "pakaian"   : False,
            "makanan"   : False,
            "musik"     : False,
            "senjata"   : False
        }

        consequents_flag = {
            "rumah"     : False,
            "pakaian"   : False,
            "makanan"   : False,
            "musik"     : False,
            "senjata"   : False
        }

        for antecedents_item in row['antecedents']:
            key = antecedents_item.split("_")[0].lower()
            antecedents_flag[key] = True

        for consequents_item in row['consequents']:
            key = consequents_item.split("_")[0].lower()
            consequents_flag[key] = True
        
        for category in base_category:
            if(antecedents_flag[category] and consequents_flag[category]):
                tf = tf.append(row)
    
    tf = tf.reset_index(drop=True)
    for index, row in tf.iterrows():
        # Create array to store the value of current list
        base = []
        clear = True
        # print("INDEX NUMBER :", index)
        for antecedents_item in row['antecedents']:
            base.append(antecedents_item)

        for consequents_item in row['consequents']:
            base.append(consequents_item)
        
        # print("BASE :",base)
        for i, r in tf.iloc[index+1:].iterrows():
            comparator = []

            for antecedents_item in r['antecedents']:
                comparator.append(antecedents_item)

            for consequents_item in r['consequents']:
                comparator.append(consequents_item)

            # print(comparator, "==", set(base) == set(comparator))
            if(set(base) == set(comparator)):
                clear = False

        if(clear):
            ff = ff.append(row)
    
    # print("SOURCE ###########################")
    # print(tf.sort_values(by='lift', ascending=False).reset_index(drop=True))
    # print("OUTPUT ###########################")
    # print(ff.sort_values(by='lift', ascending=False).reset_index(drop=True))
    ff = ff.sort_values(by='lift', ascending=False).reset_index(drop=True)
    return ff

def getNotAccessedContent(min_sp, min_conf):
    df = getSuggestion(min_sp, min_conf)
    antecedents_item = []
    consequents_item = []

    base_item_tracker = {
        "rumah" : {
            "Rumah_Aceh"        : False,
            "Rumah_Asmat"       : False,
            "Rumah_Sunda"       : False,
            "Rumah_Bali"        : False,
            "Rumah_Dayak"       : False,
            "Rumah_Toraja"      : False
        },
        "pakaian" : {
            'Pakaian_Aceh'      : False,
            'Pakaian_Asmat'     : False,
            'Pakaian_Bali'      : False,
            'Pakaian_Dayak'     : False,
            'Pakaian_Sunda'     : False,
            'Pakaian_Toraja'    : False
        },
        "makanan" : {                
            'Makanan_Aceh' : False,
            'Makanan_Asmat': False,
            'Makanan_Bali' : False,
            'Makanan_Dayak': False,
            'Makanan_Sunda': False,
            'Makanan_Toraja': False
        },
        "musik" : {
            'Musik_Aceh' : False,
            'Musik_Asmat': False,
            'Musik_Bali': False,
            'Musik_Dayak': False,
            'Musik_Sunda': False,
            'Musik_Toraja': False
        },
        "senjata" : {
            'Senjata_Aceh': False,
            'Senjata_Asmat': False,
            'Senjata_Bali': False,
            'Senjata_Dayak': False,
            'Senjata_Sunda': False,
            'Senjata_Toraja': False
        }
    }

    for row_item in df['antecedents']:
        for real_item in row_item:
            antecedents_item.append(real_item)
    
    for row_item in df['consequents']:
        for real_item in row_item:
            consequents_item.append(real_item)

    for category in base_category:
        for x in antecedents_item:
            key = x.split("_")[0].lower()
            base_item_tracker[key][x] = True
        
        for x in consequents_item:
            key = x.split("_")[0].lower()
            base_item_tracker[key][x] = True

    return base_item_tracker

def getFinalSuggestion(min_sp, min_conf):
    df = getSuggestion(min_sp, min_conf)
    df = df.sort_values(by='confidence', ascending=False)
    finalSet = {
        'rumah' : [],
        'pakaian' : [],
        'makanan' : [],
        'musik' : [],
        'senjata' : [],
    }

    finalSetFlag = {
        'rumah' : False,
        'pakaian' : False,
        'makanan' : False,
        'musik' : False,
        'senjata' : False,
    }

    for index, row in df.iterrows():
        for row_item in row['antecedents']:
            key = row_item.split("_")[0].lower()
            if not containsName(finalSet[key], row_item) :
                finalSet[key].append({
                    'item_name' : row_item,
                    'confidence' : row['confidence'],
                })
                finalSetFlag[key] = True

        for row_item in row['consequents']:
            key = row_item.split("_")[0].lower()
            if not containsName(finalSet[key], row_item) :
                finalSet[key].append({
                    'item_name' : row_item,
                    'confidence' : row['confidence'],
                })
                finalSetFlag[key] = True

    # Insert not included item
    for key in base_item:
        for item in base_item[key]:
            cont_key = False

            key = item.split("_")[0].lower()
            if  not containsName(finalSet[key], item) :
                if  finalSetFlag[key] == True:
                    tempset = []
                    for item in base_item[key]: 
                        if  not containsName(finalSet[key], item) : 
                            tempset.append({
                                'item_name' : item,
                                'confidence' : 0,
                            })
                    
                    finalSet[key] = finalSet[key]+(tempset)

                    cont_key = True
                else :
                    finalSet[key].append({
                        'item_name' : item,
                        'confidence' : 0,
                    })

            if(cont_key): break

    # pprint.pprint(finalSet)
    return finalSet

def containsName(list, filter):
    for x in list:
        if x['item_name'] == filter:        
            return True
    return False

if __name__ == "__main__":
    app.run(debug=True)
    