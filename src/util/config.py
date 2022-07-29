import configparser


SOURCES = ['dailykos', 'hpo', 'cnn', 'wpo', 'nyt', 'usatoday', 'ap', 'hill', 'wat',
    'fox', 'breitbart']

SOURCE_to_IND = {'dailykos': 0, 'hpo': 1, 'cnn': 2, 'wpo': 3, 'nyt': 4,
    'usatoday': 5, 'ap': 6, 'hill': 7, 'wat':8, 'fox': 9, 'breitbart': 10}

SOURCE_to_IDEO = {'dailykos': -1, 'hpo': -1, 'cnn': -1, 'wpo': -1, 'nyt': -1,
    'usatoday': 0, 'ap': 0, 'hill': 0, 'wat': 1, 'fox': 1, 'breitbart': 1}

LEFT_SOURCES = {0, 1, 2, 3, 4}
CENTER_SOURCES = {5, 6, 7}
RIGHT_SOURCES = {8, 9, 10}

DOWNSTREAM_TASKS = ['allsides_chronological', 'congress_speech', 'ytb_user', 'ytb_user_comment',
    'hyper_partisan', 'TVtranscript', 'basil_sentiments', 'twitter_full',
    'vast', 'semeval', 'directed_sentiment', 'basil_article_stance']
