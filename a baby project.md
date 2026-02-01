```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv("anime.csv")
```


```python
df.head()
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
      <th>anime_id</th>
      <th>name</th>
      <th>genre</th>
      <th>type</th>
      <th>episodes</th>
      <th>rating</th>
      <th>members</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32281</td>
      <td>Kimi no Na wa.</td>
      <td>Drama, Romance, School, Supernatural</td>
      <td>Movie</td>
      <td>1</td>
      <td>9.37</td>
      <td>200630</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5114</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>Action, Adventure, Drama, Fantasy, Magic, Mili...</td>
      <td>TV</td>
      <td>64</td>
      <td>9.26</td>
      <td>793665</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28977</td>
      <td>Gintama°</td>
      <td>Action, Comedy, Historical, Parody, Samurai, S...</td>
      <td>TV</td>
      <td>51</td>
      <td>9.25</td>
      <td>114262</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9253</td>
      <td>Steins;Gate</td>
      <td>Sci-Fi, Thriller</td>
      <td>TV</td>
      <td>24</td>
      <td>9.17</td>
      <td>673572</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9969</td>
      <td>Gintama&amp;#039;</td>
      <td>Action, Comedy, Historical, Parody, Samurai, S...</td>
      <td>TV</td>
      <td>51</td>
      <td>9.16</td>
      <td>151266</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info
```




    <bound method DataFrame.info of        anime_id                                               name  \
    0         32281                                     Kimi no Na wa.   
    1          5114                   Fullmetal Alchemist: Brotherhood   
    2         28977                                           Gintama°   
    3          9253                                        Steins;Gate   
    4          9969                                      Gintama&#039;   
    ...         ...                                                ...   
    12289      9316       Toushindai My Lover: Minami tai Mecha-Minami   
    12290      5543                                        Under World   
    12291      5621                     Violence Gekiga David no Hoshi   
    12292      6133  Violence Gekiga Shin David no Hoshi: Inma Dens...   
    12293     26081                   Yasuji no Pornorama: Yacchimae!!   
    
                                                       genre   type episodes  \
    0                   Drama, Romance, School, Supernatural  Movie        1   
    1      Action, Adventure, Drama, Fantasy, Magic, Mili...     TV       64   
    2      Action, Comedy, Historical, Parody, Samurai, S...     TV       51   
    3                                       Sci-Fi, Thriller     TV       24   
    4      Action, Comedy, Historical, Parody, Samurai, S...     TV       51   
    ...                                                  ...    ...      ...   
    12289                                             Hentai    OVA        1   
    12290                                             Hentai    OVA        1   
    12291                                             Hentai    OVA        4   
    12292                                             Hentai    OVA        1   
    12293                                             Hentai  Movie        1   
    
           rating  members  
    0        9.37   200630  
    1        9.26   793665  
    2        9.25   114262  
    3        9.17   673572  
    4        9.16   151266  
    ...       ...      ...  
    12289    4.15      211  
    12290    4.28      183  
    12291    4.88      219  
    12292    4.98      175  
    12293    5.46      142  
    
    [12294 rows x 7 columns]>




```python
df.isna().sum()
```




    anime_id      0
    name          0
    genre        62
    type         25
    episodes      0
    rating      230
    members       0
    dtype: int64




```python
# several column contain missing information, such as genre, type, and rating.
```


```python
df.describe(include = "all")
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
      <th>anime_id</th>
      <th>name</th>
      <th>genre</th>
      <th>type</th>
      <th>episodes</th>
      <th>rating</th>
      <th>members</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12294.000000</td>
      <td>12294</td>
      <td>12232</td>
      <td>12269</td>
      <td>12294</td>
      <td>12064.000000</td>
      <td>1.229400e+04</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>12292</td>
      <td>3264</td>
      <td>6</td>
      <td>187</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>Shi Wan Ge Leng Xiaohua</td>
      <td>Hentai</td>
      <td>TV</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>2</td>
      <td>823</td>
      <td>3787</td>
      <td>5677</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14058.221653</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.473902</td>
      <td>1.807134e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11455.294701</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.026746</td>
      <td>5.482068e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.670000</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3484.250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.880000</td>
      <td>2.250000e+02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10260.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.570000</td>
      <td>1.550000e+03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>24794.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.180000</td>
      <td>9.437000e+03</td>
    </tr>
    <tr>
      <th>max</th>
      <td>34527.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.000000</td>
      <td>1.013917e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# summary statistics for numerical columns, will including the mean, minimum, maximum, and data spread.

```

# Graph 1



```python
plt.figure()
plt.hist(df["rating"], bins=20)
plt.xlabel("Rating")
plt.ylabel("Number of Anime")
plt.title("Distribution of Anime Ratings")
plt.show()
```


    
![png](output_9_0.png)
    



```python
# in this graph it shows how anime ratings are spread across the entire dataset.  
```

# Graph 2


```python
df["type"].value_counts().plot(kind="bar")
plt.xlabel("Anime Type")
plt.ylabel("Count")
plt.title("Anime Count by Type")
plt.show()
```


    
![png](output_12_0.png)
    



```python
# in this bar chart  it shows how many anime exist in each category.  
```

# Graph 3


```python
avg_rating = df.groupby("type")["rating"].mean()

avg_rating.plot(kind="bar")
plt.xlabel("Anime Type")
plt.ylabel("Average Rating")
plt.title("Average Rating by Anime Type")
plt.show()
```


    
![png](output_15_0.png)
    



```python
# in this graph compares the average rating for each anime type.  
```
