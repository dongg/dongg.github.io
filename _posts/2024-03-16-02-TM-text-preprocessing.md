## Text Preprocessing

* 파이썬 텍스트 전처리 요약
    * [Getting started with Text Preprocessing](https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing)
    * [텍스트 자료 분석, 동국대 김진석교수](http://bigdata.dongguk.ac.kr/lectures/TextMining/_book/index.html)

###  주의사항

* 자료형태: 
    * 텍스트 자료는 보통 라인단위로 입력되고, 라인은 string임. 
    * 자료형태가 py(기본)로 처리할때는 문자열 리스트, pd는 문자열 Series, np는 문자열 array임 
    * 어떤 형태로 자료가 저장되어 있느냐 따라 처리내용이 달라짐 
* py의 문자열 리스트인 경우
    * py의 주요 함수는 str을 처리하게 되어 있음. 
    * str리스트를 처리하려면 개별 str을 처리하는 함수를 만들고 `for, [for], map`로 리스트에 적용해야 함
* pd의 문자열 Series인 경우
    * pd.Series는 원소가 문자열일 경우 `str`속성을 가지고, `str`속성은 문자열 메쏘드를 가짐
    * 문자열 Series에 직접 문자열 메쏘드를 적용할 수 있고 결과도 Series로 반환됨 (R과 유사한 방식)
    
    
### 필요한 패키지


패키지이름|용도|특이사항
:---------|:--------------------|:----------------------
`string`|문자열 관련 기본 패키지|`string.punctuation`
`re`|정규표현식 패키지|
`nltk`|Natural Language ToolKit. 자연어처리 패키지|stemmer, lemmatizer, stopwords
`pyspell`|영어 철자 교정| `!pip install pyspell`
`konlpy`|한글 자연어처리 패키지|


```python
# !pip install lxml
# !pip install nltk
import nltk
# nltk.download() # 최초 import시 실행
import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import pandas as pd
```

### string 요약

* string_manip.Rmd 참고




```python
# 문자열을 공백으로 분리 
s = 'This is it!'
tknwrd = s.split()
tknch = list(s)
print('s=', s, '=> s.split() => to the list of tkn words ', tknwrd)
print('s=', s, '=> list(s) => to the list of tkn char ', tknch)
```

    s= This is it! => s.split() => to the list of tkn words  ['This', 'is', 'it!']
    s= This is it! => list(s) => to the list of tkn char  ['T', 'h', 'i', 's', ' ', 'i', 's', ' ', 'i', 't', '!']
    


```python
idiom = ['Hit the sack ! :-) elon.musk@spacex.com', 
         'Heard it through the grapevine :-(',
         '<p>Once in a blue moon</p>', 
         'The devil is in the detail http://goo.gl', 
         'Character is destiny queen@royal.gov.uk',     # Heraclitus, George Eliot, John McCain
         'Come rain or shine ',
         '<a href="http://teacherstalking.com">Haste makes waiste</a>',
         'On cloud nine https://weather.com']
meaning = ['Go to sleep', 
           'Heard through gossip',
           'Rarely', 
           'Looks good but there are problems', 
           'destiny is not a predetermined outside force, but his own character', 
           'No matter what',
           'Make mistakes if rush', 
           'Very happy']
kr =['자러 간다','풍문으로 들었소','그럴리가 없다','악마는 디테일에 있다', '성격이 운명이다', 
     '비가 오나 해가 떠나', '서두르면 실수한다', '너무 행복 !']

# pandas
DF = pd.DataFrame({'idiom':idiom, 'meaning':meaning, 'kr':kr}) 
DF
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>idiom</th>
      <th>meaning</th>
      <th>kr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hit the sack ! :-) elon.musk@spacex.com</td>
      <td>Go to sleep</td>
      <td>자러 간다</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Heard it through the grapevine :-(</td>
      <td>Heard through gossip</td>
      <td>풍문으로 들었소</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;p&gt;Once in a blue moon&lt;/p&gt;</td>
      <td>Rarely</td>
      <td>그럴리가 없다</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The devil is in the detail http://goo.gl</td>
      <td>Looks good but there are problems</td>
      <td>악마는 디테일에 있다</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Character is destiny queen@royal.gov.uk</td>
      <td>destiny is not a predetermined outside force, ...</td>
      <td>성격이 운명이다</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Come rain or shine</td>
      <td>No matter what</td>
      <td>비가 오나 해가 떠나</td>
    </tr>
    <tr>
      <th>6</th>
      <td>&lt;a href="http://teacherstalking.com"&gt;Haste mak...</td>
      <td>Make mistakes if rush</td>
      <td>서두르면 실수한다</td>
    </tr>
    <tr>
      <th>7</th>
      <td>On cloud nine https://weather.com</td>
      <td>Very happy</td>
      <td>너무 행복 !</td>
    </tr>
  </tbody>
</table>
</div>




```python
lststr = idiom
```


```python
lststr[0].split()  
```




    ['Hit', 'the', 'sack', '!', ':-)', 'elon.musk@spacex.com']




```python
# 주의 lststr.split()은 오류. list는 split메쏘드가 없음 
# for나 list comprehension, map으로 처리해야 함 map이 더 빠른 듯
# https://www.youtube.com/watch?v=hNW6Tbp59HQ&ab_channel=BrendanMetcalfe
for t in lststr[:2]:
    print(t.split())  
print([t.split() for t in lststr[:2]])
print(list(map(lambda x: x.split(), lststr[:2])))
```

    ['Hit', 'the', 'sack', '!', ':-)', 'elon.musk@spacex.com']
    ['Heard', 'it', 'through', 'the', 'grapevine', ':-(']
    [['Hit', 'the', 'sack', '!', ':-)', 'elon.musk@spacex.com'], ['Heard', 'it', 'through', 'the', 'grapevine', ':-(']]
    [['Hit', 'the', 'sack', '!', ':-)', 'elon.musk@spacex.com'], ['Heard', 'it', 'through', 'the', 'grapevine', ':-(']]
    


```python
lststr[0].lower()  
```




    'hit the sack ! :-) elon.musk@spacex.com'




```python
print([t.lower() for t in lststr[:2]])
print(list(map(lambda x: x.lower(), lststr[:2])))
```

    ['hit the sack ! :-) elon.musk@spacex.com', 'heard it through the grapevine :-(']
    ['hit the sack ! :-) elon.musk@spacex.com', 'heard it through the grapevine :-(']
    


```python
# DF['line'].split() 오류. Series에는 split이 없음
# Series나 Index 는 문자일때 str속성(접근자)을 가지고 R:stringr과 유사한 메쏘드를 사용할 수 있음.
# for나 list comprehesion 없이 R 처럼 사용가능
DF['idiom'].str.split()
```




    0       [Hit, the, sack, !, :-), elon.musk@spacex.com]
    1            [Heard, it, through, the, grapevine, :-(]
    2                     [<p>Once, in, a, blue, moon</p>]
    3     [The, devil, is, in, the, detail, http://goo.gl]
    4         [Character, is, destiny, queen@royal.gov.uk]
    5                              [Come, rain, or, shine]
    6    [<a, href="http://teacherstalking.com">Haste, ...
    7               [On, cloud, nine, https://weather.com]
    Name: idiom, dtype: object



### [구두점 제거]( https://blog.enterprisedna.co/python-remove-punctuation-from-string/)

* 구두점 문자: 
    * `string`에 `string.punctuation`에 정의되어 있음
    * 32자 ASIICODE 33~126 사이값을 가짐


```python
# on string.punctuation : 총 32자 ASIICODE 33~126 사이값을 가짐
import string
TBL = pd.DataFrame({'punctuation':list(string.punctuation),
                    'ascii':      list(map(ord, string.punctuation))})# [ord(ch) for ch in string.punctuation]
TBL.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>punctuation</th>
      <th>ascii</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>!</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>$</td>
      <td>36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>%</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>



* 구두점 제거 1: `replace`


```python
# 방법1: replace로 한 글자씩 제거 ?
print(lststr[0].replace('!', '') )
print([s.replace('!', '') for s in lststr[:2]])
```

    Hit the sack  :-) elon.musk@spacex.com
    ['Hit the sack  :-) elon.musk@spacex.com', 'Heard it through the grapevine :-(']
    

* 구두점 제거 2: 정규표현식 `re`


```python
import re
import string

print(string.punctuation)            # string에 저장된 구두점 문자
print(re.escape(string.punctuation)) # 구두점 문자를 escape(backslash 붙이기) 

# re.compile('pat')  
# re.compile('[%s]' % re.escape(string.punctuation))
r = re.compile('[%s]' % re.escape(string.punctuation))
r.search(lststr[0]) # '[%s]' % re.escape(string.punctuation))
r.sub('', lststr[0]), r.sub('', lststr[1])
```

    !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    !"\#\$%\&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~
    




    ('Hit the sack   elonmuskspacexcom', 'Heard it through the grapevine ')




```python
def rm_punctuation(s):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', s)

print(rm_punctuation(lststr[0]))
```

    Hit the sack   elonmuskspacexcom
    


```python
# lststr.apply(lambda x: rm_punctuation(x))
print(list(map(rm_punctuation, lststr[:2])))  
[rm_punctuation(s) for s in lststr[:2]] 
```

    ['Hit the sack   elonmuskspacexcom', 'Heard it through the grapevine ']
    




    ['Hit the sack   elonmuskspacexcom', 'Heard it through the grapevine ']




```python
# 치환표. 일반 문자는 그대로 유지하되 (x='', y='', z=string.punctuation) string.punction는 제거하는 dict
str.maketrans('','', string.punctuation)  # ''를 ''로 치환, string.punctuation은 제거 
```




    {33: None,
     34: None,
     35: None,
     36: None,
     37: None,
     38: None,
     39: None,
     40: None,
     41: None,
     42: None,
     43: None,
     44: None,
     45: None,
     46: None,
     47: None,
     58: None,
     59: None,
     60: None,
     61: None,
     62: None,
     63: None,
     64: None,
     91: None,
     92: None,
     93: None,
     94: None,
     95: None,
     96: None,
     123: None,
     124: None,
     125: None,
     126: None}




```python
lststr[0].translate(str.maketrans('','', string.punctuation))
```




    'Hit the sack   elonmuskspacexcom'




```python
# 2. How to Use Translate() Method to Delete Punctuation
def rm_punctuation2(s):
    trns = str.maketrans('','', string.punctuation)
    # trns:{아스키코드:대체할 내용} 
    return s.translate(trns)

print(rm_punctuation2(lststr[1]))
list(map(rm_punctuation2, lststr))
```

    Heard it through the grapevine 
    




    ['Hit the sack   elonmuskspacexcom',
     'Heard it through the grapevine ',
     'pOnce in a blue moonp',
     'The devil is in the detail httpgoogl',
     'Character is destiny queenroyalgovuk',
     'Come rain or shine ',
     'a hrefhttpteacherstalkingcomHaste makes waistea',
     'On cloud nine httpsweathercom']




```python
DF['idiom'].apply(lambda s: rm_punctuation(s))
```




    0                   Hit the sack   elonmuskspacexcom
    1                    Heard it through the grapevine 
    2                              pOnce in a blue moonp
    3               The devil is in the detail httpgoogl
    4               Character is destiny queenroyalgovuk
    5                                Come rain or shine 
    6    a hrefhttpteacherstalkingcomHaste makes waistea
    7                      On cloud nine httpsweathercom
    Name: idiom, dtype: object




```python
DF['idiom'].apply(lambda s: rm_punctuation2(s))
```




    0                   Hit the sack   elonmuskspacexcom
    1                    Heard it through the grapevine 
    2                              pOnce in a blue moonp
    3               The devil is in the detail httpgoogl
    4               Character is destiny queenroyalgovuk
    5                                Come rain or shine 
    6    a hrefhttpteacherstalkingcomHaste makes waistea
    7                      On cloud nine httpsweathercom
    Name: idiom, dtype: object



### 불용어 제거

* `nltk`에 `stopwords`에 179개의 불용어가 정의되어 있음


```python
# Stopwords
from nltk.corpus import stopwords
print(len(stopwords.words('english')))
':'.join(stopwords.words('english'))
```

    179
    




    "i:me:my:myself:we:our:ours:ourselves:you:you're:you've:you'll:you'd:your:yours:yourself:yourselves:he:him:his:himself:she:she's:her:hers:herself:it:it's:its:itself:they:them:their:theirs:themselves:what:which:who:whom:this:that:that'll:these:those:am:is:are:was:were:be:been:being:have:has:had:having:do:does:did:doing:a:an:the:and:but:if:or:because:as:until:while:of:at:by:for:with:about:against:between:into:through:during:before:after:above:below:to:from:up:down:in:out:on:off:over:under:again:further:then:once:here:there:when:where:why:how:all:any:both:each:few:more:most:other:some:such:no:nor:not:only:own:same:so:than:too:very:s:t:can:will:just:don:don't:should:should've:now:d:ll:m:o:re:ve:y:ain:aren:aren't:couldn:couldn't:didn:didn't:doesn:doesn't:hadn:hadn't:hasn:hasn't:haven:haven't:isn:isn't:ma:mightn:mightn't:mustn:mustn't:needn:needn't:shan:shan't:shouldn:shouldn't:wasn:wasn't:weren:weren't:won:won't:wouldn:wouldn't"




```python
# 불용어 지정: split으로 토큰화후 제거하고 다시 join해서 복구 
STOPWORDS = set(stopwords.words('english'))
def rm_stopwords(s):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(s).split() if word not in STOPWORDS])

print(rm_stopwords(lststr[0]))
list(map(rm_stopwords, lststr))
# df["text_wo_stop"] = df["text_wo_punct"].apply(lambda text: remove_stopwords(text))
# df.head()
```

    Hit sack ! :-) elon.musk@spacex.com
    




    ['Hit sack ! :-) elon.musk@spacex.com',
     'Heard grapevine :-(',
     '<p>Once blue moon</p>',
     'The devil detail http://goo.gl',
     'Character destiny queen@royal.gov.uk',
     'Come rain shine',
     '<a href="http://teacherstalking.com">Haste makes waiste</a>',
     'On cloud nine https://weather.com']




```python
# 불용어 지정: split으로 토큰화후 제거하고 다시 join해서 복구 
DF['idiom'].apply(lambda s: rm_stopwords(s))
```




    0                  Hit sack ! :-) elon.musk@spacex.com
    1                                  Heard grapevine :-(
    2                                <p>Once blue moon</p>
    3                       The devil detail http://goo.gl
    4                 Character destiny queen@royal.gov.uk
    5                                      Come rain shine
    6    <a href="http://teacherstalking.com">Haste mak...
    7                    On cloud nine https://weather.com
    Name: idiom, dtype: object



### Stemming 어간추출: 

* 예: walks, walking => walk, console,  consoling => consol (not proper word)
* Stemming algorithms: PorterStemmer(영어만 가능), SnowballStemmer
* 적용시 처리된 단어가 사전단어가 아닐 수 있음
* 토큰화 필요 (DL4NLP참고할 것). 
    * 1) 파이썬 split으로 분리하거나
    * 2) nltk의 word_tokenize(Don't=> Do n't로 분리), WordPunctTokenizer(Don't=> Don ' t로 분리)
    * 3) tf.keras의 text_to_word_sequence (자동 소문자화, 구두점 제거, don't => don't; 아포스트로피 유지 )


```python
from nltk.stem.porter import PorterStemmer

PStemmer = PorterStemmer()
def stem_words(s):
    return ' '.join([PStemmer.stem(word) for word in s.split()])

# df["text_stemmed"] = df["text"].apply(lambda text: stem_words(text))
# df.head()
print(stem_words(lststr[0]))
list(map(stem_words, lststr)) # Problematic: once => onc, character=>charact, haste=>hast, waste=>wast
```

    hit the sack ! :-) elon.musk@spacex.com
    




    ['hit the sack ! :-) elon.musk@spacex.com',
     'heard it through the grapevin :-(',
     '<p>onc in a blue moon</p>',
     'the devil is in the detail http://goo.gl',
     'charact is destini queen@royal.gov.uk',
     'come rain or shine',
     '<a href="http://teacherstalking.com">hast make waiste</a>',
     'on cloud nine https://weather.com']




```python
from nltk.stem.snowball import SnowballStemmer
# SnowballStemmer.languages : 지원하는 언어
SStemmer = SnowballStemmer('english')
def stem_words2(s):
    return ' '.join([SStemmer.stem(word) for word in s.split()])

# df["text_stemmed"] = df["text"].apply(lambda text: stem_words(text))
# df.head()
print(stem_words2(lststr[1]))
list(map(stem_words2, lststr)) # Problematic: once => onc, character=>charact, haste=>hast, waste=>wast
```

    heard it through the grapevin :-(
    




    ['hit the sack ! :-) elon.musk@spacex.com',
     'heard it through the grapevin :-(',
     '<p>onc in a blue moon</p>',
     'the devil is in the detail http://goo.gl',
     'charact is destini queen@royal.gov.uk',
     'come rain or shine',
     '<a href="http://teacherstalking.com">hast make waiste</a>',
     'on cloud nine https://weather.com']




```python
# 어간 추출
DF['idiom'].apply(lambda s: stem_words(s))  # 자동 소문자화함
```




    0              hit the sack ! :-) elon.musk@spacex.com
    1                    heard it through the grapevin :-(
    2                            <p>onc in a blue moon</p>
    3             the devil is in the detail http://goo.gl
    4                charact is destini queen@royal.gov.uk
    5                                   come rain or shine
    6    <a href="http://teacherstalking.com">hast make...
    7                    on cloud nine https://weather.com
    Name: idiom, dtype: object



### Lemmatization 표제어(기본사전형) 추출
* 어간추출과 유사하지만 온전한 단어(어근, root word = lemma)로 추출. 오래 걸림
* POS tagging하므로 n(oun), v(erb) 등을 지정할 수 있음 
* NLTK WordNetLemmatizer 


```python
from nltk.stem import WordNetLemmatizer
WNLemmatizer = WordNetLemmatizer()
def lemmatize_words(s):
    return ' '.join([WNLemmatizer.lemmatize(word, 'v') for word in s.split()])
    # WNLemmatizer.lemmatize(word, ['v|n|'])

# df["text_lemmatized"] = df["text"].apply(lambda text: lemmatize_words(text))
# df.head()
print(lemmatize_words(lststr[1]))
list(map(lemmatize_words, lststr)) # makes => make
```

    Heard it through the grapevine :-(
    




    ['Hit the sack ! :-) elon.musk@spacex.com',
     'Heard it through the grapevine :-(',
     '<p>Once in a blue moon</p>',
     'The devil be in the detail http://goo.gl',
     'Character be destiny queen@royal.gov.uk',
     'Come rain or shine',
     '<a href="http://teacherstalking.com">Haste make waiste</a>',
     'On cloud nine https://weather.com']




```python
# 표제어 추출
DF['idiom'].apply(lambda s: lemmatize_words(s))
```




    0              Hit the sack ! :-) elon.musk@spacex.com
    1                   Heard it through the grapevine :-(
    2                           <p>Once in a blue moon</p>
    3             The devil be in the detail http://goo.gl
    4              Character be destiny queen@royal.gov.uk
    5                                   Come rain or shine
    6    <a href="http://teacherstalking.com">Haste mak...
    7                    On cloud nine https://weather.com
    Name: idiom, dtype: object



### Removal of URLs


```python
# Removal of URLs
def rm_urls(s):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', s)

# print(rm_urls(lststr[1]))
print(rm_urls('Please refer to http://google.co.kr for details'))
list(map(rm_urls, lststr))
```

    Please refer to  for details
    




    ['Hit the sack ! :-) elon.musk@spacex.com',
     'Heard it through the grapevine :-(',
     '<p>Once in a blue moon</p>',
     'The devil is in the detail ',
     'Character is destiny queen@royal.gov.uk',
     'Come rain or shine ',
     '<a href=" makes waiste</a>',
     'On cloud nine ']




```python
# Removal of URLs
DF['idiom'].apply(lambda s: rm_urls(s))
```




    0    Hit the sack ! :-) elon.musk@spacex.com
    1         Heard it through the grapevine :-(
    2                 <p>Once in a blue moon</p>
    3                The devil is in the detail 
    4    Character is destiny queen@royal.gov.uk
    5                        Come rain or shine 
    6                 <a href=" makes waiste</a>
    7                             On cloud nine 
    Name: idiom, dtype: object



### Removal of HTML Tags


```python
# Removal of HTML Tags
def rm_html(s):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', s)

print(rm_html('<pre> re.compile("<.*?>") </pre>')) 
list(map(rm_html, lststr))
```

     re.compile("") 
    




    ['Hit the sack ! :-) elon.musk@spacex.com',
     'Heard it through the grapevine :-(',
     'Once in a blue moon',
     'The devil is in the detail http://goo.gl',
     'Character is destiny queen@royal.gov.uk',
     'Come rain or shine ',
     'Haste makes waiste',
     'On cloud nine https://weather.com']




```python
# Removal of URLs
DF['idiom'].apply(lambda s: rm_html(s))
```




    0     Hit the sack ! :-) elon.musk@spacex.com
    1          Heard it through the grapevine :-(
    2                         Once in a blue moon
    3    The devil is in the detail http://goo.gl
    4     Character is destiny queen@royal.gov.uk
    5                         Come rain or shine 
    6                          Haste makes waiste
    7           On cloud nine https://weather.com
    Name: idiom, dtype: object



### Spelling Correction
* `!pip install pyspellchecker`


```python
# None 처리 때문에 수정해야 함
from spellchecker import SpellChecker

SChecker = SpellChecker()
def correct_spellings(s):
    corrected_text = []
    misspelled_words = SChecker.unknown(s.split())
    for word in s.split():
        if word in misspelled_words:
            corrected_text.append(SChecker.correction(word))
        else:
            corrected_text.append(word)
    print(corrected_text)
    corrected_text = [i for i  in corrected_text if i is not None]
    return " ".join(corrected_text)
        
text = 'speling correctin'
print(correct_spellings(text))
list(map(correct_spellings, lststr))  # 수정내용에 None이 포함되면 join이 안됨. 
```

    ['spelling', 'correcting']
    spelling correcting
    ['Hit', 'the', 'sack', '!', None, None]
    ['Heard', 'it', 'through', 'the', 'grapevine', None]
    ['<p>Once', 'in', 'a', 'blue', None]
    ['The', 'devil', 'is', 'in', 'the', 'detail', None]
    ['Character', 'is', 'destiny', None]
    ['Come', 'rain', 'or', 'shine']
    ['a', 'href="http://teacherstalking.com">Haste', 'makes', None]
    ['On', 'cloud', 'nine', None]
    




    ['Hit the sack !',
     'Heard it through the grapevine',
     '<p>Once in a blue',
     'The devil is in the detail',
     'Character is destiny',
     'Come rain or shine',
     'a href="http://teacherstalking.com">Haste makes',
     'On cloud nine']




```python
DF['idiom'].apply(lambda s: correct_spellings(s))  # 철자 오류없어서 None
```

    ['Hit', 'the', 'sack', '!', None, None]
    ['Heard', 'it', 'through', 'the', 'grapevine', None]
    ['<p>Once', 'in', 'a', 'blue', None]
    ['The', 'devil', 'is', 'in', 'the', 'detail', None]
    ['Character', 'is', 'destiny', None]
    ['Come', 'rain', 'or', 'shine']
    ['a', 'href="http://teacherstalking.com">Haste', 'makes', None]
    ['On', 'cloud', 'nine', None]
    




    0                                     Hit the sack !
    1                     Heard it through the grapevine
    2                                  <p>Once in a blue
    3                         The devil is in the detail
    4                               Character is destiny
    5                                 Come rain or shine
    6    a href="http://teacherstalking.com">Haste makes
    7                                      On cloud nine
    Name: idiom, dtype: object




```python
poem = pd.read_csv('D:/GHUB//TM2023/개여울.txt')
poem
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>line</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>당신은 무슨 일로 그리합니까</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홀로 이 개여울에 주저앉아서</td>
    </tr>
    <tr>
      <th>2</th>
      <td>파릇한 풀포기가 돋아 나오고</td>
    </tr>
    <tr>
      <th>3</th>
      <td>잔물이 봄바람에 헤적일 때에</td>
    </tr>
    <tr>
      <th>4</th>
      <td>가도 아주 가지는 않노라시던</td>
    </tr>
    <tr>
      <th>5</th>
      <td>그런 약속이 있었겠지요</td>
    </tr>
    <tr>
      <th>6</th>
      <td>날마다 개여울에 나와 앉아서</td>
    </tr>
    <tr>
      <th>7</th>
      <td>하염없이 무엇을 생각합니다</td>
    </tr>
    <tr>
      <th>8</th>
      <td>가도 아주 가지는 않노라심은</td>
    </tr>
    <tr>
      <th>9</th>
      <td>굳이 잊지 말라는 부탁인지요</td>
    </tr>
    <tr>
      <th>10</th>
      <td>가도 아주 가지는 않노라시던</td>
    </tr>
    <tr>
      <th>11</th>
      <td>그런 약속이 있었겠지요</td>
    </tr>
    <tr>
      <th>12</th>
      <td>날마다 개여울에 나와 앉아서</td>
    </tr>
    <tr>
      <th>13</th>
      <td>하염없이 무엇을 생각합니다</td>
    </tr>
    <tr>
      <th>14</th>
      <td>가도 아주 가지는 않노라심은</td>
    </tr>
    <tr>
      <th>15</th>
      <td>굳이 잊지 말라는 부탁인지요</td>
    </tr>
  </tbody>
</table>
</div>




```python
# with open('D:/GHUB/TM2023/peterpan.txt') as file:
#  print(file.readlines())
poem = pd.read_csv('D:/GHUB/TM2023/peterpan.txt', delimiter=':')
poem
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>line</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I won't grow up</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I don't want to go to school</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Just to learn to be a puppet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>And recite a silly rule</td>
    </tr>
    <tr>
      <th>4</th>
      <td>If growing up means it would be</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Beneath my dignity to climb a tree</td>
    </tr>
    <tr>
      <th>6</th>
      <td>I won't grow up, won't grow up, won't grow up</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Not me.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>I won't grow up</td>
    </tr>
    <tr>
      <th>9</th>
      <td>I don't want to wear a tie</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Or a serious expression</td>
    </tr>
    <tr>
      <th>11</th>
      <td>In the middle of July</td>
    </tr>
    <tr>
      <th>12</th>
      <td>And if it means that I must prepare</td>
    </tr>
    <tr>
      <th>13</th>
      <td>To shoulder burdens with a worried air</td>
    </tr>
    <tr>
      <th>14</th>
      <td>I'll never grow up, won't grow up, never grow up</td>
    </tr>
    <tr>
      <th>15</th>
      <td>So there</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Never gonna be a man</td>
    </tr>
    <tr>
      <th>17</th>
      <td>I won't!</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Like to see somebody try</td>
    </tr>
    <tr>
      <th>19</th>
      <td>And make me!</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Anyone who wants to try</td>
    </tr>
    <tr>
      <th>21</th>
      <td>And make me turn into a man</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Catch me if you can</td>
    </tr>
    <tr>
      <th>23</th>
      <td>I won't grow up</td>
    </tr>
    <tr>
      <th>24</th>
      <td>I don't want to wear a tie</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Or a serious expression</td>
    </tr>
    <tr>
      <th>26</th>
      <td>To try to act just like a guy</td>
    </tr>
    <tr>
      <th>27</th>
      <td>'Cause growing up's awfuller than</td>
    </tr>
    <tr>
      <th>28</th>
      <td>All of the awful things that's ever been</td>
    </tr>
    <tr>
      <th>29</th>
      <td>I won't grow up, won't grow up, won't grow up</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Again</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Growing up's awfuller than</td>
    </tr>
    <tr>
      <th>32</th>
      <td>All of the awful thing that's ever been</td>
    </tr>
    <tr>
      <th>33</th>
      <td>I won't grow up, never grow up, won't grow up</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Again.</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Not me.</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Not I.</td>
    </tr>
  </tbody>
</table>
</div>




```python
poem.info() # poem[['line']]
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37 entries, 0 to 36
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   line    37 non-null     object
    dtypes: object(1)
    memory usage: 424.0+ bytes
    


```python
DF = poem
DF['txt'] = DF['line']

# 알파벳만 남기기
DF['txt'] = DF['txt'].str.replace("[^a-zA-Z]", " ")
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
DF['txt'] = DF['txt'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# 전체 단어에 대한 소문자 변환 poem['line'].str.lower() 도 됨 Series를 str로 바꾼뒤 lower
DF['txt'] = DF['txt'].apply(lambda x: x.lower())
DF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>line</th>
      <th>txt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I won't grow up</td>
      <td>won't grow</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I don't want to go to school</td>
      <td>don't want school</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Just to learn to be a puppet</td>
      <td>just learn puppet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>And recite a silly rule</td>
      <td>recite silly rule</td>
    </tr>
    <tr>
      <th>4</th>
      <td>If growing up means it would be</td>
      <td>growing means would</td>
    </tr>
  </tbody>
</table>
</div>




```python
# NLTK 불용어처리
from nltk.corpus import stopwords
':'.join(stopwords.words('english'))
```




    "i:me:my:myself:we:our:ours:ourselves:you:you're:you've:you'll:you'd:your:yours:yourself:yourselves:he:him:his:himself:she:she's:her:hers:herself:it:it's:its:itself:they:them:their:theirs:themselves:what:which:who:whom:this:that:that'll:these:those:am:is:are:was:were:be:been:being:have:has:had:having:do:does:did:doing:a:an:the:and:but:if:or:because:as:until:while:of:at:by:for:with:about:against:between:into:through:during:before:after:above:below:to:from:up:down:in:out:on:off:over:under:again:further:then:once:here:there:when:where:why:how:all:any:both:each:few:more:most:other:some:such:no:nor:not:only:own:same:so:than:too:very:s:t:can:will:just:don:don't:should:should've:now:d:ll:m:o:re:ve:y:ain:aren:aren't:couldn:couldn't:didn:didn't:doesn:doesn't:hadn:hadn't:hasn:hasn't:haven:haven't:isn:isn't:ma:mightn:mightn't:mustn:mustn't:needn:needn't:shan:shan't:shouldn:shouldn't:wasn:wasn't:weren:weren't:won:won't:wouldn:wouldn't"




```python
# 불용어를 제거합니다.
# vs 
STOPWORDS = set(stopwords.words('english'))
def rm_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

poem['txt'] = poem['txt'].apply(lambda text: rm_stopwords(text))
poem.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>line</th>
      <th>txt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I won't grow up</td>
      <td>grow</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I don't want to go to school</td>
      <td>want school</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Just to learn to be a puppet</td>
      <td>learn puppet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>And recite a silly rule</td>
      <td>recite silly rule</td>
    </tr>
    <tr>
      <th>4</th>
      <td>If growing up means it would be</td>
      <td>growing means would</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 
```


```python
poem['txt'] = poem['line'].str.lower()
poem.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>line</th>
      <th>txt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I won't grow up</td>
      <td>i won't grow up</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I don't want to go to school</td>
      <td>i don't want to go to school</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Just to learn to be a puppet</td>
      <td>just to learn to be a puppet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>And recite a silly rule</td>
      <td>and recite a silly rule</td>
    </tr>
    <tr>
      <th>4</th>
      <td>If growing up means it would be</td>
      <td>if growing up means it would be</td>
    </tr>
  </tbody>
</table>
</div>




```python
import string  
PUNCT_TO_REMOVE = string.punctuation
PUNCT_TO_REMOVE
```




    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'




```python
def rm_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
poem['txt'] = poem['txt'].apply(lambda text: rm_punctuation(text))
# df["text_wo_punct"] = df["text"].apply(lambda text: remove_punctuation(text))
# df.head()
poem.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>line</th>
      <th>txt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I won't grow up</td>
      <td>i wont grow up</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I don't want to go to school</td>
      <td>i dont want to go to school</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Just to learn to be a puppet</td>
      <td>just to learn to be a puppet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>And recite a silly rule</td>
      <td>and recite a silly rule</td>
    </tr>
    <tr>
      <th>4</th>
      <td>If growing up means it would be</td>
      <td>if growing up means it would be</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove stopwords
# stopwords in nltk
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
```




    "i, me, my, myself, we, our, ours, ourselves, you, you're, you've, you'll, you'd, your, yours, yourself, yourselves, he, him, his, himself, she, she's, her, hers, herself, it, it's, its, itself, they, them, their, theirs, themselves, what, which, who, whom, this, that, that'll, these, those, am, is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, a, an, the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some, such, no, nor, not, only, own, same, so, than, too, very, s, t, can, will, just, don, don't, should, should've, now, d, ll, m, o, re, ve, y, ain, aren, aren't, couldn, couldn't, didn, didn't, doesn, doesn't, hadn, hadn't, hasn, hasn't, haven, haven't, isn, isn't, ma, mightn, mightn't, mustn, mustn't, needn, needn't, shan, shan't, shouldn, shouldn't, wasn, wasn't, weren, weren't, won, won't, wouldn, wouldn't"




```python
STOPWORDS = set(stopwords.words('english'))
def rm_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

poem['txt'] = poem['txt'].apply(lambda text: rm_stopwords(text))
poem.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>line</th>
      <th>txt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I won't grow up</td>
      <td>wont grow</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I don't want to go to school</td>
      <td>dont want go school</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Just to learn to be a puppet</td>
      <td>learn puppet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>And recite a silly rule</td>
      <td>recite silly rule</td>
    </tr>
    <tr>
      <th>4</th>
      <td>If growing up means it would be</td>
      <td>growing means would</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove frequent words
from collections import Counter
cnt = Counter()
for text in poem['txt'].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)
```




    [('grow', 15),
     ('wont', 13),
     ('never', 4),
     ('dont', 3),
     ('want', 3),
     ('growing', 3),
     ('try', 3),
     ('means', 2),
     ('wear', 2),
     ('tie', 2)]




```python
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def rm_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

poem['txt'] = poem['txt'].apply(lambda text: rm_freqwords(text))
poem.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>line</th>
      <th>txt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I won't grow up</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>I don't want to go to school</td>
      <td>go school</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Just to learn to be a puppet</td>
      <td>learn puppet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>And recite a silly rule</td>
      <td>recite silly rule</td>
    </tr>
    <tr>
      <th>4</th>
      <td>If growing up means it would be</td>
      <td>would</td>
    </tr>
  </tbody>
</table>
</div>




```python
# pandas
DF = pd.DataFrame({'idiom':idiom, 'meaning':meaning, 'kr':kr}) 
DF


DF['idiom'].apply(lambda s: s.lower())

```




    0              hit the sack ! :-) elon.musk@spacex.com
    1                   heard it through the grapevine :-(
    2                           <p>once in a blue moon</p>
    3             the devil is in the detail http://goo.gl
    4              character is destiny queen@royal.gov.uk
    5                                  come rain or shine 
    6    <a href="http://teacherstalking.com">haste mak...
    7                    on cloud nine https://weather.com
    Name: idiom, dtype: object



### 예제: 한 번에 처리

* 유의: 전처리 순서에 따라 결과가 많이 바뀜. 예: HTML태그 제거후 URL제거, URL제거후 HTML제거 등


```python
# pandas
DF = pd.DataFrame({'idiom':idiom, 'meaning':meaning, 'kr':kr}) 

# 소문자화
DF['idiom'] = DF['idiom'].apply(lambda s: s.lower())

# Removal of HTML Tags
def rm_html(s):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', s)
DF['idiom'] = DF['idiom'].apply(lambda s: rm_html(s))



# Removal of URLs
def rm_urls(s):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', s)
DF['idiom'] = DF['idiom'].apply(lambda s: rm_urls(s))


# Removal of emails
def rm_emails(s):
    email_pattern = re.compile(r'\S*@\S*\s?')
    return email_pattern.sub(r'', s)
DF['idiom'] = DF['idiom'].apply(lambda s: rm_emails(s))


# 구두점 제거 
# on string.punctuation : 총 32자 ASIICODE 33~126 사이값을 가짐
import string

def rm_punctuation2(s):
    trns = str.maketrans('','', string.punctuation)
    # trns:{아스키코드:대체할 내용} 
    return s.translate(trns)
DF['idiom'] = DF['idiom'].apply(lambda s: rm_punctuation2(s))

# 불용어 지정: split으로 토큰화후 제거하고 다시 join해서 복구 
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def rm_stopwords(s):
    return " ".join([word for word in str(s).split() if word not in STOPWORDS])
DF['idiom'] = DF['idiom'].apply(lambda s: rm_stopwords(s))

# 표제화
from nltk.stem import WordNetLemmatizer
WNLemmatizer = WordNetLemmatizer()
def lemmatize_words(s):
    return ' '.join([WNLemmatizer.lemmatize(word, 'v') for word in s.split()])
DF['idiom'] = DF['idiom'].apply(lambda s: lemmatize_words(s))
DF
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>idiom</th>
      <th>meaning</th>
      <th>kr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hit sack</td>
      <td>Go to sleep</td>
      <td>자러 간다</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hear grapevine</td>
      <td>Heard through gossip</td>
      <td>풍문으로 들었소</td>
    </tr>
    <tr>
      <th>2</th>
      <td>blue moon</td>
      <td>Rarely</td>
      <td>그럴리가 없다</td>
    </tr>
    <tr>
      <th>3</th>
      <td>devil detail</td>
      <td>Looks good but there are problems</td>
      <td>악마는 디테일에 있다</td>
    </tr>
    <tr>
      <th>4</th>
      <td>character destiny</td>
      <td>destiny is not a predetermined outside force, ...</td>
      <td>성격이 운명이다</td>
    </tr>
    <tr>
      <th>5</th>
      <td>come rain shine</td>
      <td>No matter what</td>
      <td>비가 오나 해가 떠나</td>
    </tr>
    <tr>
      <th>6</th>
      <td>haste make waiste</td>
      <td>Make mistakes if rush</td>
      <td>서두르면 실수한다</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cloud nine</td>
      <td>Very happy</td>
      <td>너무 행복 !</td>
    </tr>
  </tbody>
</table>
</div>




```python
docs = DF['idiom']
docs
```




    0             hit sack
    1       hear grapevine
    2            blue moon
    3         devil detail
    4    character destiny
    5      come rain shine
    6    haste make waiste
    7           cloud nine
    Name: idiom, dtype: object




```python
from sklearn.feature_extraction.text import CountVectorizer
Tcount = CountVectorizer()
Tcount.fit(docs)
Tcount.get_feature_names_out(), Tcount.vocabulary_  # 단어집합
```




    (array(['blue', 'character', 'cloud', 'come', 'destiny', 'detail', 'devil',
            'grapevine', 'haste', 'hear', 'hit', 'make', 'moon', 'nine',
            'rain', 'sack', 'shine', 'waiste'], dtype=object),
     {'hit': 10,
      'sack': 15,
      'hear': 9,
      'grapevine': 7,
      'blue': 0,
      'moon': 12,
      'devil': 6,
      'detail': 5,
      'character': 1,
      'destiny': 4,
      'come': 3,
      'rain': 14,
      'shine': 16,
      'haste': 8,
      'make': 11,
      'waiste': 17,
      'cloud': 2,
      'nine': 13})




```python
DTTF = Tcount.fit_transform(docs)
DTTF
```




    <8x18 sparse matrix of type '<class 'numpy.int64'>'
    	with 18 stored elements in Compressed Sparse Row format>




```python
DTTF.toarray()
```




    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
          dtype=int64)




```python
from sklearn.feature_extraction.text import TfidfVectorizer
Ttfidf = TfidfVectorizer(stop_words='english', max_features=1000)
Ttfidf.fit(docs)
Ttfidf.get_feature_names_out(), Ttfidf.vocabulary_  # 단어집합
```




    (array(['blue', 'character', 'cloud', 'come', 'destiny', 'devil',
            'grapevine', 'haste', 'hear', 'hit', 'make', 'moon', 'rain',
            'sack', 'shine', 'waiste'], dtype=object),
     {'hit': 9,
      'sack': 13,
      'hear': 8,
      'grapevine': 6,
      'blue': 0,
      'moon': 11,
      'devil': 5,
      'character': 1,
      'destiny': 4,
      'come': 3,
      'rain': 12,
      'shine': 14,
      'haste': 7,
      'make': 10,
      'waiste': 15,
      'cloud': 2})




```python
DTTFIDF = Ttfidf.transform(docs)
pd.DataFrame(DTTFIDF.toarray(), columns=Ttfidf.get_feature_names_out())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>blue</th>
      <th>character</th>
      <th>cloud</th>
      <th>come</th>
      <th>destiny</th>
      <th>devil</th>
      <th>grapevine</th>
      <th>haste</th>
      <th>hear</th>
      <th>hit</th>
      <th>make</th>
      <th>moon</th>
      <th>rain</th>
      <th>sack</th>
      <th>shine</th>
      <th>waiste</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.707107</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.707107</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.707107</td>
      <td>0.00000</td>
      <td>0.707107</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.707107</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.707107</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.707107</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.707107</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.57735</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.57735</td>
      <td>0.000000</td>
      <td>0.57735</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.57735</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.57735</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.57735</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
</div>



## TF를 이용한 임베딩


```python

```
