def get_city(img_dir):
    # finds city name after the second to last '/' character
    # assumes directory ends with '/' as well
    # e.g. restofpath/cityname/ -> cityname  b
    return img_dir[img_dir.rfind('/',0,len(img_dir)-1)+1 : len(img_dir)-1]

def get_base_dir(img_dir):
    # gets the base directory without the city name. 
    # Assumes that it is of form 'base_dir/cityname/'
    return img_dir[:img_dir.rfind('/', 0, len(img_dir)-1)] + '/'

def piecewise_linear(score, base_cutoffs, city_cutoffs):
    group = 1 # starting at 1, because it has to be at least in the 0 to 1 index
    while (score < city_cutoffs[group] and group<4):
        group+=1
    
    city_diff = city_cutoffs[group-1] - city_cutoffs[group]
    base_diff = base_cutoffs[group-1] - base_cutoffs[group]

    new_score = (score-city_cutoffs[group]) * base_diff/city_diff + base_cutoffs[group]

    return new_score

def get_cutoffs(df):
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    first_index = [df.where(df['trueskill_category']==i).first_valid_index() for i in range(4)]
    last_index  = [df.where(df['trueskill_category']==i).last_valid_index() for i in range(4)]
    cutoff_01 = (float(df.iloc[[last_index[0]]]['score']) + float(df.iloc[[first_index[1]]]['score']))/2
    cutoff_12 = (float(df.iloc[[last_index[1]]]['score']) + float(df.iloc[[first_index[2]]]['score']))/2
    cutoff_23 = (float(df.iloc[[last_index[2]]]['score']) + float(df.iloc[[first_index[3]]]['score']))/2
    #print(first_index)
    #print(last_index)
    cutoffs = [df.iloc[0]['score'], cutoff_01, cutoff_12, cutoff_23, df.iloc[len(df)-1]['score']]
    #print(df)
    return cutoffs
