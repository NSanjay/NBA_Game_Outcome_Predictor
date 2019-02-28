import getData
from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(0)

ofeatures = ['homeAway','home_team_home_record_pct','away_team_home_record_pct','W_PCT','home_team_current_win_percentage'
    ,'away_team_current_win_percentage','home_team_current_standing_series','away_team_current_standing_series']
def load_data():
    overalldf = None
    for team in getData.teamToIndex.keys():
        #df = getData.load_teamBoxScoresBetweenYears(team,2000,2017)
        df = getData.load_custom_data(team,2015,'BOS')

        if overalldf is None:
            overalldf = df
        else:
            overalldf.append(df)
        df.to_csv('./dataset/{0}.csv'.format(team))
        break
    return df

def main():
    with open('./dataset/data2.csv','w',encoding='utf-8') as f:
        for team in getData.teamToIndex.keys():
            print("team:::",team)
            df = getData.load_customized_data(team,2016,2017)
            print(df.head())
            print(list(df))
            df.to_csv(f,mode='a',header=True)
    #print("df:::",load_data())

def winLossToBool(truth):
    for i in range(len(truth)):
        if truth[i] == 'W':
            truth[i] = 1
        else:
            truth[i] = 0
    return truth

def addOtherFeaturesToPlayingTeams(dataframe, feature, additionalfeatures):
    features = dataframe[feature].values.tolist()
    otherfeatures = dataframe[additionalfeatures].values
    #print("other:::",otherfeatures)
    for i in range(len(features)):
        featurelist = features[i].tolist()
        #print("featurelist::",featurelist)
        for j in range(len(otherfeatures[i])):
            if(otherfeatures[i][j] != 'nan'):
                featurelist.append(otherfeatures[i][j])
            else:
                print('nan')
                featurelist.append(0)
        features[i] = featurelist
    return features

def trainSVM(train):
    #clf = svm.SVC(kernel='rbf')
    clf = linear_model.LogisticRegression(random_state=0, C=3.0)
    #clf = ensemble.RandomForestClassifier(n_estimators=10, max_depth=2, criterion="entropy", random_state=0)
    #clf = GaussianNB()
    features = addOtherFeaturesToPlayingTeams(train,'playingTeams',ofeatures)
    #print("train_features:::",features)
    truth = winLossToBool(train['WL'].values.tolist())
    clf.fit(features,truth)
    return clf

def testSVM(clf, test):
    truth = winLossToBool(test['WL'].values.tolist())
    features = addOtherFeaturesToPlayingTeams(test,'playingTeams',ofeatures)
##    print(features)
    correct = 0
    total = len(test)
    for t in range(len(test)):
        if(truth[t] == 'nan'):
            print('nan')
        if(truth[t] == clf.predict([features[t]])):
           correct+=1
    return(correct/float(total))

def add_features(df):
    home = []
    tpArray = []
    dateweightArray = []
    #dates = df['GAME_DATE'].values
    i = 0
    #print(df.head())
    for index, row in df.iterrows():
        teamsPlaying = np.zeros(len(getData.teamToIndex))
        game_id = row["Game_ID"]
        #print("id:::",game_id)
        matchup = row['MATCHUP']
        teams = matchup.split(" ")
        if '@' in matchup:
            home.append(0)
        else:
            home.append(1)
        t1Index = getData.teamToIndex[teams[0]]
        t2Index = getData.teamToIndex[teams[2]]
        teamsPlaying[t1Index] = 1
        teamsPlaying[t2Index] = 1
        tpArray.append(teamsPlaying)

        date = row['GAME_DATE'].split("-")
        year = "{:}{:}".format(20,date[2])
        #print ("date::",year)
        dateweightArray.append(0.9 ** (2017 - int(date[2]) + 1))
    ha = pd.Series(home)
    # print("ha:::",ha)
    tp = pd.Series(tpArray)
    # print("tp:::", tp)
    dw = pd.Series(dateweightArray)
    df = df.assign(homeAway=ha.values)
    df = df.assign(playingTeams=tp.values)
    df = df.assign(dateWeight=dw.values)
    return df

def doTest():
    num_iterations = 20
    X=[i for i in range(1,num_iterations+1)]
    Y=[]
    trainError = []
    df = pd.read_csv('./dataset/data.csv', header=0)
    totalAccuracy = 0
    totalTrainAccuracy = 0
    for _ in range(num_iterations):
        #get train, test
        train, test = train_test_split(df, test_size=0.2,shuffle=True)
        train = add_features(train)
        test = add_features(test)
        #train svm
        clf = trainSVM(train)
        #test svm
        totalAccuracy = testSVM(clf, test)
        totalTrainAccuracy = testSVM(clf,train)
        print("Accuracy:::", totalAccuracy)
        print("TrainAccuracy:::", totalTrainAccuracy)
        trainError.append(totalTrainAccuracy)
        Y.append(totalAccuracy)

    print("TEST ACCURACY: %f" % (np.mean(Y)))
    print("TRAIN ACCURACY: %f" % (np.mean(trainError)))
    plt.plot(X, Y, marker='o')
    #plt.xticks(X, teams, rotation=45, fontsize=10)
    plt.ylim(ymin=0.3, ymax=1)

    #plt.plot([0, 31], [0.5, 0.5], color='red')
    #plt.grid(True)
    plt.show()

    def main():
        doSVM = True
        doRFC = True
        doLR = True
        doNB = True
        modelTrainAcc = {}
        modelTestAcc = {}
        if (doSVM):
            print("SVM Classifier")
            Model = 'SVM'
            X, TestAcc, TrainAcc, teams = dealWithModels(Model)
            modelTestAcc[Model] = np.mean(TestAcc)
            modelTrainAcc[Model] = np.mean(TrainAcc)
            plt.figure(1)
            plt.title("SVM Classifier")
            test_acc = plt.plot(X, TestAcc, marker='o', color='blue', label='Test Accuracy')
            train_acc = plt.plot(X, TrainAcc, marker='o', color='green', label='Train Accuracy')
            # plt.legend((test_acc, train_acc), (, 'Test Accuracy'))
            plt.xticks(X, teams, rotation=45, fontsize=10)
            plt.ylim(ymin=0.3, ymax=1)
            plt.plot([0, 31], [0.5, 0.5], color='red')
            plt.grid(True)
        if (doRFC):
            print("Random Forest Classifier")
            Model = 'RFC'
            X, TestAcc, TrainAcc, teams = dealWithModels(Model)
            modelTestAcc[Model] = np.mean(TestAcc)
            modelTrainAcc[Model] = np.mean(TrainAcc)
            plt.figure(2)
            plt.title("Random Forest Classifier")
            test_acc = plt.plot(X, TestAcc, marker='o', color='blue', label='Test Accuracy')
            train_acc = plt.plot(X, TrainAcc, marker='o', color='green', label='Train Accuracy')
            # plt.legend((test_acc, train_acc), ('Test Accuracy', 'Test Accuracy'))
            plt.xticks(X, teams, rotation=45, fontsize=10)
            plt.ylim(ymin=0.3, ymax=1)
            plt.plot([0, 31], [0.5, 0.5], color='red')
            plt.grid(True)
        if (doLR):
            print("Logistic Regression")
            Model = 'LR'
            X, TestAcc, TrainAcc, teams = dealWithModels(Model)
            modelTestAcc[Model] = np.mean(TestAcc)
            modelTrainAcc[Model] = np.mean(TrainAcc)
            plt.figure(3)
            plt.title("Logistic Regression Classifier")
            test_acc = plt.plot(X, TestAcc, marker='o', color='blue', label='Test Accuracy')
            train_acc = plt.plot(X, TrainAcc, marker='o', color='green', label='Train Accuracy')
            # plt.legend((test_acc, train_acc), ('Test Accuracy', 'Test Accuracy'))
            plt.xticks(X, teams, rotation=45, fontsize=10)
            plt.ylim(ymin=0.3, ymax=1)
            plt.plot([0, 31], [0.5, 0.5], color='red')
            plt.grid(True)
        if (doNB):
            print("Naive Bayes")
            Model = 'NB'
            X, TestAcc, TrainAcc, teams = dealWithModels(Model)
            modelTestAcc[Model] = np.mean(TestAcc)
            modelTrainAcc[Model] = np.mean(TrainAcc)
            plt.figure(4)
            plt.title("Naive Bayes Classifier")
            test_acc = plt.plot(X, TestAcc, marker='o', color='blue', label='Test Accuracy')
            train_acc = plt.plot(X, TrainAcc, marker='o', color='green', label='Train Accuracy')
            # plt.legend((test_acc, train_acc), ('Test Accuracy', 'Test Accuracy'))
            plt.xticks(X, teams, rotation=45, fontsize=10)
            plt.ylim(ymin=0.3, ymax=1)
            plt.plot([0, 31], [0.5, 0.5], color='red')
            plt.grid(True)
        plt.figure(5)
        barWidth = 0.25

        # set height of bar
        bars1 = modelTrainAcc.values()
        bars2 = modelTestAcc.values()

        # Set position of bar on X axis
        r1 = np.arange(len(bars1))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]

        # Make the plot
        plt.bar(r1, bars1, color='#1E90FF', width=barWidth, edgecolor='white', label='Train Accuracy')
        plt.bar(r2, bars2, color='#8B008B', width=barWidth, edgecolor='white', label='Test Accuracy')

        # Add xticks on the middle of the group bars
        plt.xlabel('group', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(bars1))],
                   ['SVM Classifier', 'Random Forest Classifier', 'Logistic Regression Classifier',
                    'Naive Bayes Classifier'])

        # Create legend & Show graphic
        plt.legend()
        plt.show()

if __name__ == '__main__':
    #main()
    doTest()