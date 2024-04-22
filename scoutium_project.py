############################################################
# SCOUTIUM - Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma
############################################################

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from matplotlib import pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=Warning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz. Adım 2: Okutmuş olduğumuz
# csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz. ("task_response_id", 'match_id', 'evaluator_id'
# "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)

def load_csvs():
    df_attributes = pd.read_csv("MachineLearning/machine_learning/Case_Scoutium/scoutium_attributes.csv", sep=";")
    df_potential_labels = pd.read_csv("MachineLearning/machine_learning/Case_Scoutium/scoutium_potential_labels.csv",
                                      sep=";")
    df_return = pd.merge(df_attributes, df_potential_labels, how="inner",
                         on=["task_response_id", "match_id", "evaluator_id", "player_id"])
    return df_return


df = load_csvs()


def quick_look(dataframe):
    print("############ SHAPE ############")
    print(dataframe.shape)
    print("############ DTYPES ############")
    print(dataframe.dtypes)
    print("############ IS NULL ############")
    print(dataframe.isnull().sum().sort_values(ascending=False))
    print("############ DESCRIBE ############")
    print(dataframe.describe().T)
    print("############ TARGET VALUE COUNTS ############")
    print(dataframe["potential_label"].value_counts())
    print("############ HEAD ############")
    print(dataframe.head())
    print("############ TAIL ############")
    print(dataframe.tail())


quick_look(df)

# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df = df.loc[df["position_id"] != 1, :]
print(df.loc[df["position_id"] == 1, :])

# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm
# verisetinin %1'ini oluşturur)

round(df["potential_label"].value_counts() * 100 / len(df), 2)
df = df.loc[df["potential_label"] != "below_average", :]
print(df[df["potential_label"] == "below_average"])

# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz.
# Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.

# Adım 5.1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
# “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
df = df.pivot_table(index=["player_id", "position_id", "potential_label"],
                    columns=["attribute_id"],
                    values="attribute_value")
df.head()
print(df.columns)
print(df.dtypes)
print(df.index)

# Adım 5.2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.
df.reset_index(inplace=True)
df.head()
print(df.dtypes)
float_columns = [col for col in df.columns if df[col].dtype == "float64"]
print(df[4322].head())
df.columns = [str(col) if df[col].dtype == "float64" else col for col in df.columns]
print(df["4322"].head())


# Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal
# olarak ifade ediniz

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    print(labelencoder.inverse_transform([0, 1]))
    return dataframe


df["potential_label"].unique()
label_encoder(df, "potential_label")
df.head()

# Adım 7: Sayısal değişken kolonlarını "num_cols" adıyla bir listeye atayınız.

print(df.dtypes)


def grab_col_names(dataframe):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   (dataframe[col].dtype not in ["O", "float64"]) & (dataframe[col].nunique() < 10)]
    num_but_cat = [col for col in num_but_cat if "potential_label" not in col]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_cols = [col for col in num_cols if (col not in num_but_cat) & (col not in ["player_id", "potential_label"])]
    return cat_cols, num_cols


cat_cols, num_cols = grab_col_names(df)

# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()

# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir
# makine öğrenmesi modeli geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

y = df["potential_label"]
X = df.drop(["player_id", "potential_label"], axis=1)


def base_models(X, y, scoring=["roc_auc"]):
    print("Base Models....")
    classifiers = [  # ('LR', LogisticRegression()),
        ('KNN', KNeighborsClassifier()),
        # ("SVC", SVC()),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier()),
        # ('Adaboost', AdaBoostClassifier()),
        ('GBM', GradientBoostingClassifier()),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ('LightGBM', LGBMClassifier()),
        # ('CatBoost', CatBoostClassifier(verbose=False))
    ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring[0]}: {round(cv_results['test_roc_auc'].mean(), 4)} ({name}) ")
        print(f"{scoring[1]}: {round(cv_results['test_f1'].mean(), 4)} ({name}) ")
        print(f"{scoring[2]}: {round(cv_results['test_precision'].mean(), 4)} ({name}) ")
        print(f"{scoring[3]}: {round(cv_results['test_recall'].mean(), 4)} ({name}) ")
        print(f"{scoring[4]}: {round(cv_results['test_accuracy'].mean(), 4)} ({name}) ")


base_models(X, y, scoring=["roc_auc", "f1", "precision", "recall", "accuracy"])

"""
roc_auc: 0.7426 (KNN) 
f1: 0.4821 (KNN) 
precision: 1.0 (KNN) 
recall: 0.3226 (KNN) 
accuracy: 0.8597 (KNN) 
#
roc_auc: 0.7077 (CART) 
f1: 0.547 (CART) 
precision: 0.5807 (CART) 
recall: 0.5361 (CART) 
accuracy: 0.8084 (CART) 
#
roc_auc: 0.8903 (RF) 
f1: 0.5702 (RF) 
precision: 0.8583 (RF) 
recall: 0.4288 (RF) 
accuracy: 0.8671 (RF) 
#
roc_auc: 0.8559 (GBM) 
f1: 0.5912 (GBM) 
precision: 0.7027 (GBM) 
recall: 0.5175 (GBM) 
accuracy: 0.8524 (GBM) 
#
roc_auc: 0.8441 (XGBoost) 
f1: 0.6037 (XGBoost) 
precision: 0.6904 (XGBoost) 
recall: 0.5526 (XGBoost) 
accuracy: 0.8488 (XGBoost) 
#
roc_auc: 0.8588 (LightGBM) 
f1: 0.5788 (LightGBM) 
precision: 0.6599 (LightGBM) 
recall: 0.5175 (LightGBM) 
accuracy: 0.845 (LightGBM) 
"""

lgbm_model = LGBMClassifier()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8709
cv_results['test_f1'].mean()
# 0.6475
cv_results['test_roc_auc'].mean()
# 0.8829

lgbm_model.get_params()
# colsample_bytree=1.0, learning_rate=0.1, max_depth=-1, n_estimators=100

# Hyperparameter optimization
params = {"colsample_bytree": [1.0, 0.5, 1.5],
          "learning_rate": [0.1, 0.01, 0.005],
          "max_depth": [-1, 10, 50, -5],
          "n_estimators": [100, 300, 200, 450]}

best_params = GridSearchCV(lgbm_model, params, cv=5, verbose=True, n_jobs=-1).fit(X, y)

lgbm_final = lgbm_model.set_params(**best_params.best_params_, random_state=17).fit(X, y)
lgbm_final.get_params()
# 'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': -1, 'n_estimators': 300

# 5-Fold Cross Validation
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8856
cv_results['test_f1'].mean()
# 0.6527
cv_results['test_roc_auc'].mean()
# 0.8814

# Test Set w/Confusion Matrix & Roc Auc
# Confusion matrix for y_pred:
y_pred = lgbm_final.predict(X)

# Classification report
print(classification_report(y, y_pred))

# Confusion matrix
cm = confusion_matrix(y, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=df["potential_label"].unique(),
            yticklabels=df["potential_label"].unique())
plt.xlabel("Tahmin Edilen Etiket")
plt.ylabel("Gerçek Etiket")
plt.title("Hata Matrisi")
plt.show()

# Recall : (47 / 56) = 0,839  | Bir modelin ne kadar iyi pozitif örnekleri tespit ettiğini ölçer.
# Precision : (47 / 48) = 0,979 | Bir modelin pozitif olarak tahmin ettiği örneklerin ne kadarının gerçekten pozitif olduğunu ölçer.
# Accuracy : (214 + 47) / (214 + 9 + 47 + 1) = 0,96


# AUC for y_prob:
y_prob = lgbm_final.predict_proba(X)[:, 1]
# AUC
roc_auc_score(y, y_prob)
# 0.9960


# Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak
# özelliklerin sıralamasını çizdiriniz.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_final, X)

# Prediction using model
random_player = X.sample(1)
rp_index = random_player.index.values[0]
print(df[df.index == rp_index]["potential_label"])
lgbm_final.predict(random_player)





