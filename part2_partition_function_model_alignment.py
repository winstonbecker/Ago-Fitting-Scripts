# Import modules
import scipy
import numpy as np
import pandas as pd
import sys
import os
import lmfit
from lmfit import minimize, Parameters, Parameter, report_fit
import copy
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)

#########################################################################
# function definitions

def initialize_alignment_params(guide, model_params = None):
    # Function to initialize the model parameters and package them in a dict
    # Outputs:
    # parvals--dict containing dynamic programming parameters
    parvals = dict()
    # define initially bound parameters
    position_penalties = pd.DataFrame(columns = ['A', 'C', 'G', 'U'], index = range(0,21,1))
    #########################################################################
    if model_params is None:
        # Initialize Double flips (min penalty: 0 kcal/mol; max penalty: 7 kcal/mol; initial penalty is double the average initial flip penalty at that position)
        parvals['seed_opening_bulge_guide'] = 1.4
        parvals['central_opening_bulge_guide'] = 0.9
        parvals['supp_opening_bulge_guide'] = 1.1
        parvals['seed_second_nt_bulge_guide'] = 0.9   
        parvals['central_second_nt_bulge_guide'] = 0.9
        parvals['supp_second_nt_bulge_guide'] = 0.9
        parvals['seed_opening_bulge_target'] = 1.2   
        parvals['central_opening_bulge_target'] = 0.6
        parvals['supp_opening_bulge_target'] = 0.7
        parvals['seed_second_nt_bulge_target'] = 0.2   
        parvals['central_second_nt_bulge_target'] = 0.2
        parvals['supp_second_nt_bulge_target'] = 0.2
        parvals['initiation_penalty'] = 0.5
        # cant make the bulge penalties too different because they represent relatively large regions
        #########################################################################
        if guide == 'let7':
            parvals['seed_opening_bulge_guide'] = 1.6
            parvals['central_opening_bulge_guide'] = 1
            parvals['supp_opening_bulge_guide'] = 1.1
            parvals['seed_second_nt_bulge_guide'] = 1   
            parvals['central_second_nt_bulge_guide'] = 1
            parvals['supp_second_nt_bulge_guide'] = 1
            parvals['seed_opening_bulge_target'] = 1.4 
            parvals['central_opening_bulge_target'] = 0.6
            parvals['supp_opening_bulge_target'] = 0.8
            parvals['seed_second_nt_bulge_target'] = 0.2   
            parvals['central_second_nt_bulge_target'] = 0.2
            parvals['supp_second_nt_bulge_target'] = 0.2
            parvals['initiation_penalty'] = 0.4
            position_penalties['A'] = [-1.3, 0.4, 0.4, 0.4, 0.4,-1.4, 0.3, 0.3,-0.6, 0.1, 0.1, 0.1,-0.8,-0.8, 0.3,-0.8, 0.2,-0.5, 0.2, 0.2,-0.3]
            position_penalties['C'] = [-0.5,-1.6, 0.4,-1.6,-1.5, 0.3, 0.3,-1.3, 0.1, 0.1,-0.6,-0.6, 0.3, 0.3,-0.8, 0.3, 0.2, 0.2, 0.2,-0.3, 0.2]
            position_penalties['G'] = [-0.5, 0.4, 0.4, 0.4, 0.4,-0.2, 0.3, 0.3,-0.1, 0.1, 0.1, 0.1,-0.1,-0.1, 0.3,-0.1, 0.2,-0.2, 0.2, 0.2,-0.1]
            position_penalties['U'] = [-0.5,-0.4,-1.7,-0.4,-0.3, 0.3,-1.3,-0.2, 0.1,-0.6,-0.1,-0.1, 0.3, 0.3,-0.1, 0.3,-0.6, 0.2,-0.4,-0.1, 0.2]
        else:
            parvals['seed_opening_bulge_guide'] = 1.6
            parvals['central_opening_bulge_guide'] = 1
            parvals['supp_opening_bulge_guide'] = 1.1
            parvals['seed_second_nt_bulge_guide'] = 1   
            parvals['central_second_nt_bulge_guide'] = 1
            parvals['supp_second_nt_bulge_guide'] = 1
            parvals['seed_opening_bulge_target'] = 1.4 
            parvals['central_opening_bulge_target'] = 0.6
            parvals['supp_opening_bulge_target'] = 0.8
            parvals['seed_second_nt_bulge_target'] = 0.2   
            parvals['central_second_nt_bulge_target'] = 0.2
            parvals['supp_second_nt_bulge_target'] = 0.2
            parvals['initiation_penalty'] = 0.4
            position_penalties['A'] = [-1.3, 0.4, 0.4, 0.4,-1.5,-1.4, 0.3,-1.3, 0.1, 0.1, 0.1, 0.1, 0.3,-1.2, 0.3, 0.3,-0.6, 0.2,-0.4,-0.3, 0.2]
            position_penalties['C'] = [-0.5, 0.4,-2.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.1, 0.1,-0.5, 0.1, 0.3, 0.3,-1.2, 0.3, 0.2,-0.5, 0.2, 0.2,-0.3]
            position_penalties['G'] = [-0.5,-0.4, 0.4,-2.5,-0.3,-0.3, 0.3,-0.3,-0.5, 0.1, 0.1, 0.1,-1.2,-0.2, 0.3, 0.3,-0.2, 0.2,-0.2,-0.1, 0.2]
            position_penalties['U'] = [-0.5,-2.5,-0.4, 0.4, 0.4, 0.3,-1.3, 0.3, 0.1,-0.5,-0.1,-0.5, 0.3, 0.3,-0.2,-1.2, 0.2,-0.2, 0.2, 0.1,-0.1]
    else:
        position_penalties['A'] = model_params[0:21]
        position_penalties['C'] = model_params[21:42]
        position_penalties['G'] = model_params[42:63]
        position_penalties['U'] = model_params[63:84]
        parvals['seed_opening_bulge_guide'] = model_params[84]
        parvals['central_opening_bulge_guide'] = model_params[85]
        parvals['supp_opening_bulge_guide'] = model_params[86]
        parvals['seed_second_nt_bulge_guide'] = model_params[87]  
        parvals['central_second_nt_bulge_guide'] = model_params[88]
        parvals['supp_second_nt_bulge_guide'] = model_params[89] 
        parvals['seed_opening_bulge_target'] = model_params[90]
        parvals['central_opening_bulge_target'] = model_params[91]
        parvals['supp_opening_bulge_target'] = model_params[92]
        parvals['seed_second_nt_bulge_target'] = model_params[93]
        parvals['central_second_nt_bulge_target'] = model_params[94]
        parvals['supp_second_nt_bulge_target'] = model_params[95]
        parvals['initiation_penalty'] = model_params[96]
    #parvals['initiation_penalty'] = 0.5
    # add parameters to dict
    k=0
    for i in range(0,21,1):
        for j in ['A', 'C', 'G', 'U']:
            parvals[j+str(i+1)] = position_penalties.loc[i][j]
            k = k+1
    # return the dict of params
    return parvals


def compute_score_matricies(target_sequence, guide_sequence, wt_Target, parvals):
    # target_sequence - 3' to 5'
    # guide_sequence - 5' to 3'
    # position_penalties--data frame with rows as nucleotides and columns as positions going from g1-g21
    # columns correspond to guide positions (so a 2 in origins indicates a bulge in the guide)
    # rows correspond to target positions (so a 1 in origins indicates a bulge in the target)
    # all penalties should be divided by -kbT and exponentiated
    initiation_penalty = parvals['initiation_penalty']
    matched_min = 0
    matched_min_index = [0,0]
    target_seq_len = len(target_sequence)
    # create matricies to store scores and origins of scores
    # using lists of list because pandas was slow for lookups and difficult to use things like apply and iter rows when testing different model formulaitons
    # this includes an extra row and columns of zeros so that there is something to refer to. As a result column 1 (zero indexed) refers to guide position 1 
    # and row 1 (zero indexed) refers to target position 1.
    # the dynammic programming matricies will be all caps
    MATCHED = [[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100] for i in range(target_seq_len)] # this is a 21xtarget_seq_len matrix
    MISMATCHED = [[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100] for i in range(target_seq_len)] # this is a 21xtarget_seq_len matrix
    TARGET_BULGE = [[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100] for i in range(target_seq_len)] # this is a 21xtarget_seq_len matrix
    GUIDE_BULGE = [[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100] for i in range(target_seq_len)] # this is a 21xtarget_seq_len matrix
    # initilize first row and column in the bulge matricies to 0 so that they dont get added to the partition function when a match is introduced. Instead 
    # we have a +1 term in the MATCHED computation to account for all possible dangling end which we assume to have zero energy. Could structure differently 
    # and initialize to one and remove comments (also saves on # of operations)
    # fill in row 0 and column 0 of the matched matrix with the position 1 penalties:
    for i in range(0,21,1):
        #TARGET_BULGE[0][i] = 1
        #GUIDE_BULGE[0][i] = 1
        MATCHED[0][i] = parvals[target_sequence[0]+str(i+1)]+initiation_penalty
    for i in range(0,target_seq_len,1):
        #TARGET_BULGE[i][0] = 1
        #GUIDE_BULGE[i][0] = 1
        MATCHED[i][0] = parvals[target_sequence[i]+'1']+initiation_penalty
    # fill in matricies (i is rows, j is columns)
    # populate column by column so that there is less need to look up the bulge penalties
    for j in range(1,21,1):
        # set bulge values depending on how far you are into the guide:
        if j >0 and j<8:
            opening_bulge_guide = parvals['seed_opening_bulge_guide']
            second_nt_bulge_guide = parvals['seed_second_nt_bulge_guide']
        elif j>=8 and j<13:
            opening_bulge_guide = parvals['central_opening_bulge_guide']
            second_nt_bulge_guide = parvals['central_second_nt_bulge_guide']
        elif j>=13 and j<19:
            opening_bulge_guide = parvals['supp_opening_bulge_guide']
            second_nt_bulge_guide = parvals['supp_second_nt_bulge_guide']
        elif j >= 19:
            opening_bulge_guide = 0.0
            second_nt_bulge_guide = 0.0
        # in target bulge matrix the current guide position is already bound
        if j >0 and j<7:
            opening_bulge_target = parvals['seed_opening_bulge_target']
            second_nt_bulge_target = parvals['seed_second_nt_bulge_target']
        elif j>=7 and j<12:
            opening_bulge_target = parvals['central_opening_bulge_target']
            second_nt_bulge_target = parvals['central_second_nt_bulge_target']
        elif j>=12 and j<18:
            opening_bulge_target = parvals['supp_opening_bulge_target']
            second_nt_bulge_target = parvals['supp_second_nt_bulge_target']
        elif j >= 18:
            opening_bulge_target = 0.0
            second_nt_bulge_target = 0.0
        for i in range(1,len(target_sequence),1):
            if target_sequence[i] == wt_Target[j] or (target_sequence[i] == 'G' and wt_Target[j] == 'A')  or (target_sequence[i] == 'U' and wt_Target[j] == 'C'):
                MATCHED[i][j] = parvals[target_sequence[i]+str(j+1)]+min(MATCHED[i-1][j-1],GUIDE_BULGE[i-1][j-1]+initiation_penalty,TARGET_BULGE[i-1][j-1]+initiation_penalty,initiation_penalty,MISMATCHED[i-1][j-1]+initiation_penalty) # include initiation penalty here?
                if MATCHED[i][j]<matched_min:
                    matched_min = MATCHED[i][j]
                    matched_min_index = [i,j]
            else:
                MISMATCHED[i][j] = parvals[target_sequence[i]+str(j+1)]+min(MATCHED[i-1][j-1],MISMATCHED[i-1][j-1])
            TARGET_BULGE[i][j] = min(opening_bulge_target+MATCHED[i-1][j],opening_bulge_target+MISMATCHED[i-1][j],second_nt_bulge_target+TARGET_BULGE[i-1][j])
            GUIDE_BULGE[i][j] = min(opening_bulge_guide+MATCHED[i][j-1],second_nt_bulge_guide+GUIDE_BULGE[i][j-1],opening_bulge_guide+MISMATCHED[i][j-1])
    #print pd.DataFrame(MATCHED)
    #print pd.DataFrame(MISMATCHED)
    #print pd.DataFrame(TARGET_BULGE)
    #print pd.DataFrame(GUIDE_BULGE)
    return MATCHED, MISMATCHED, TARGET_BULGE, GUIDE_BULGE, matched_min, matched_min_index


def get_match_origin(MATCHED, MISMATCHED, TARGET_BULGE, GUIDE_BULGE, i,j, parvals):
    ''' function to find the origin of a match matrix element
    returns the matrix it came from and the new index'''
    min_value = min(MATCHED[i-1][j-1],GUIDE_BULGE[i-1][j-1]+parvals['initiation_penalty'],TARGET_BULGE[i-1][j-1]+parvals['initiation_penalty'],parvals['initiation_penalty'],MISMATCHED[i-1][j-1])
    if MATCHED[i-1][j-1] == min_value:
        current_index = [i-1,j-1]
        mut_type = 'matched'
    elif GUIDE_BULGE[i-1][j-1]+parvals['initiation_penalty'] == min_value:
        current_index = [i-1,j-1]
        mut_type = 'g_bulged'
    elif TARGET_BULGE[i-1][j-1]+parvals['initiation_penalty'] == min_value:
        current_index = [i-1,j-1]
        mut_type = 't_bulged'
    elif MISMATCHED[i-1][j-1] == min_value:
        current_index = [i-1,j-1]
        mut_type = 'mismatched'
    else:
        #current_index = [0,0]
        mut_type = 'unbound'
        current_index = [i,j]
    return mut_type, current_index


def get_mismatch_origin(MATCHED, MISMATCHED, i,j, parvals):
    ''' function to find the origin of a mismatch matrix element
    returns the matrix it came from and the new index'''
    if MATCHED[i-1][j-1]<MISMATCHED[i-1][j-1]:
        current_index = [i-1,j-1]
        mut_type = 'matched'
    else:
        current_index = [i-1,j-1]
        mut_type = 'mismatched'
    return mut_type, current_index


def get_t_bulge_origin(MATCHED, MISMATCHED, TARGET_BULGE, i,j, parvals):
    ''' function to find the origin of a target bulge matrix element
    returns the matrix it came from and the new index'''
    if j >0 and j<8:
        opening_bulge_target = parvals['seed_opening_bulge_target']
        second_nt_bulge_target = parvals['seed_second_nt_bulge_target']
    elif j>=8 and j<13:
        opening_bulge_target = parvals['central_opening_bulge_target']
        second_nt_bulge_target = parvals['central_second_nt_bulge_target']
    elif j>=13 and j<19:
        opening_bulge_target = parvals['supp_opening_bulge_target']
        second_nt_bulge_target = parvals['supp_second_nt_bulge_target']
    elif j >= 19:
        opening_bulge_target = 0.0
        second_nt_bulge_target = 0.0
    # in target bulge matrix the current guide position is already bound
    if j >0 and j<7:
        opening_bulge_target = parvals['seed_opening_bulge_target']
        second_nt_bulge_target = parvals['seed_second_nt_bulge_target']
    elif j>=7 and j<12:
        opening_bulge_target = parvals['central_opening_bulge_target']
        second_nt_bulge_target = parvals['central_second_nt_bulge_target']
    elif j>=12 and j<18:
        opening_bulge_target = parvals['supp_opening_bulge_target']
        second_nt_bulge_target = parvals['supp_second_nt_bulge_target']
    elif j >= 18:
        opening_bulge_target = 0.0
        second_nt_bulge_target = 0.0
    min_value = min(opening_bulge_target+MATCHED[i-1][j],opening_bulge_target+MISMATCHED[i-1][j],second_nt_bulge_target+TARGET_BULGE[i-1][j])
    if opening_bulge_target+MATCHED[i-1][j] == min_value:
        current_index = [i-1,j]
        mut_type = 'matched'
    elif second_nt_bulge_target+TARGET_BULGE[i-1][j] == min_value:
        current_index = [i-1,j]
        mut_type = 't_bulged'
    else:
        current_index = [i-1,j]
        mut_type = 'mismatched'
    return mut_type, current_index


def get_g_bulge_origin(MATCHED, MISMATCHED, GUIDE_BULGE, i,j, parvals):
    ''' function to find the origin of a guide bulge matrix element
    returns the matrix it came from and the new index'''
    if j >0 and j<8:
        opening_bulge_guide = parvals['seed_opening_bulge_guide']
        second_nt_bulge_guide = parvals['seed_second_nt_bulge_guide']
    elif j>=8 and j<13:
        opening_bulge_guide = parvals['central_opening_bulge_guide']
        second_nt_bulge_guide = parvals['central_second_nt_bulge_guide']
    elif j>=13 and j<19:
        opening_bulge_guide = parvals['supp_opening_bulge_guide']
        second_nt_bulge_guide = parvals['supp_second_nt_bulge_guide']
    elif j >= 19:
        opening_bulge_guide = 0.0
        second_nt_bulge_guide = 0.0
    min_value = min(opening_bulge_guide+MATCHED[i][j-1],second_nt_bulge_guide+GUIDE_BULGE[i][j-1],second_nt_bulge_guide+MISMATCHED[i][j-1])
    if opening_bulge_guide+MATCHED[i][j-1] == min_value:
        current_index = [i,j-1]
        mut_type = 'matched'
    elif second_nt_bulge_guide+GUIDE_BULGE[i][j-1] == min_value:
        current_index = [i,j-1]
        mut_type = 'g_bulged'
    else:
        current_index = [i,j-1]
        mut_type = 'mismatched'
    return mut_type, current_index


def traceback(MATCHED, MISMATCHED, TARGET_BULGE, GUIDE_BULGE, matched_min_index, target_sequence, guide_sequence, parvals):
    ''' function to traceback through the dynamic programming recursion matricies to find the optimal alignment'''
    if matched_min_index[1] < len(guide_sequence)-1:
        guide_seq_rep = ''.join(guide_sequence[matched_min_index[1]+1::][::-1])
    else:
        guide_seq_rep = ''
    if matched_min_index[0] < len(target_sequence)-1:
        target_seq_rep = ''.join(target_sequence[matched_min_index[0]+1::][::-1])
    else:
        target_seq_rep = ''
    seq_relationship = ''
    #matched_indicies = ['','','','','','','','','','','','','','','','','','','','','']
    #mismatched_indicies = ['','','','','','','','','','','','','','','','','','','','','']
    mut_type = 'matched'
    current_index = matched_min_index
    while current_index[1]>0:
        if mut_type == 'matched':
            guide_seq_rep = guide_seq_rep+guide_sequence[current_index[1]]
            target_seq_rep = target_seq_rep+target_sequence[current_index[0]]
            if (guide_sequence[current_index[1]] == 'G' and target_sequence[current_index[0]] == 'U') or (guide_sequence[current_index[1]] == 'U' and target_sequence[current_index[0]] == 'G'):
                seq_relationship = seq_relationship+'.'
            else:
                seq_relationship = seq_relationship+'|'
            #matched_indicies[current_index[1]] = target_sequence[current_index[0]]
            mut_type, current_index = get_match_origin(MATCHED, MISMATCHED, TARGET_BULGE, GUIDE_BULGE, current_index[0],current_index[1],parvals)
            if mut_type == 'unbound':
                if current_index[1]<=current_index[0]:
                    guide_seq_rep = guide_seq_rep+''.join(guide_sequence[0:current_index[1]][::-1])
                    target_seq_rep = target_seq_rep+''.join(target_sequence[current_index[0]-current_index[1]:current_index[0]][::-1])
                    seq_relationship = seq_relationship+''.join([' ']*(current_index[1]))
                    current_index = [current_index[0]-current_index[1],0]
                else:
                    guide_seq_rep = guide_seq_rep+''.join(guide_sequence[0:current_index[1]][::-1])
                    target_seq_rep = target_seq_rep+''.join(target_sequence[0:current_index[0]][::-1])+''.join([' ']*(current_index[1]-current_index[0]))
                    seq_relationship = seq_relationship+''.join([' ']*(current_index[1]))
                    current_index = [0,0]
        elif mut_type == 'mismatched':
            guide_seq_rep = guide_seq_rep+guide_sequence[current_index[1]]
            target_seq_rep = target_seq_rep+target_sequence[current_index[0]]
            seq_relationship = seq_relationship+'X'
            #mismatched_indicies[current_index[1]] = target_sequence[current_index[0]]
            mut_type, current_index = get_mismatch_origin(MATCHED, MISMATCHED, current_index[0],current_index[1],parvals)
        elif mut_type == 'g_bulged':
            guide_seq_rep = guide_seq_rep+guide_sequence[current_index[1]]
            target_seq_rep = target_seq_rep+'-'
            seq_relationship = seq_relationship+'-'
            mut_type, current_index = get_g_bulge_origin(MATCHED, MISMATCHED, GUIDE_BULGE, current_index[0],current_index[1],parvals)
        elif mut_type == 't_bulged':
            guide_seq_rep = guide_seq_rep+'-'
            target_seq_rep = target_seq_rep+target_sequence[current_index[0]]
            seq_relationship = seq_relationship+'-'
            mut_type, current_index = get_t_bulge_origin(MATCHED, MISMATCHED, TARGET_BULGE, current_index[0],current_index[1],parvals)
        if current_index[0] == 0:
            break
    if mut_type == 'matched':
        guide_seq_rep = guide_seq_rep+guide_sequence[current_index[1]]
        target_seq_rep = target_seq_rep+target_sequence[current_index[0]]
        if (guide_sequence[current_index[1]] == 'G' and target_sequence[current_index[0]] == 'U') or (guide_sequence[current_index[1]] == 'U' and target_sequence[current_index[0]] == 'G'):
                seq_relationship = seq_relationship+'.'
        else:
            seq_relationship = seq_relationship+'|'
    elif mut_type == 'mismatched':
        guide_seq_rep = guide_seq_rep+guide_sequence[current_index[1]]
        target_seq_rep = target_seq_rep+target_sequence[current_index[0]]
        seq_relationship = seq_relationship+'X'
    elif mut_type == 'g_bulged':
        guide_seq_rep = guide_seq_rep+guide_sequence[current_index[1]]
        target_seq_rep = target_seq_rep+'-'
        seq_relationship = seq_relationship+'-'
    elif mut_type == 't_bulged':
        guide_seq_rep = guide_seq_rep+'-'
        target_seq_rep = target_seq_rep+target_sequence[current_index[0]]
        seq_relationship = seq_relationship+'-'
    print(''.join([' ']*(current_index[0])) + guide_seq_rep[::-1])
    print(''.join([' ']*(current_index[0])) + seq_relationship[::-1])
    print(''.join(target_sequence[0:current_index[0]])+target_seq_rep[::-1])
    bound_indices = [] # first base at 3' end of the target will be 1
    guide_indicies = [np.nan]*(current_index[0])
    target_indicies = [0]*(current_index[0])
    for i in range(min(len(guide_seq_rep), len(seq_relationship))):
        if guide_seq_rep[::-1][i] in ['A', 'C', 'G', 'U']:
            if seq_relationship[::-1][i] == '|' or seq_relationship[::-1][i] == '.':
                guide_indicies.append(1)
                target_indicies.append(1)
            elif seq_relationship[::-1][i] == 'X':
                guide_indicies.append(2)
                target_indicies.append(2)
            elif seq_relationship[::-1][i] == '-':
                guide_indicies.append(0)
                target_indicies.append(np.nan)
        if guide_seq_rep[::-1][i] == '-':
            guide_indicies.append(np.nan)
            target_indicies.append(0)
    guide_indicies = guide_indicies+[0]*(len(guide_seq_rep)-len(seq_relationship))
    guide_indicies = guide_indicies + [np.nan]*(len(''.join(target_sequence[0:current_index[0]])+target_seq_rep[::-1])-len(guide_indicies))
    target_indicies = target_indicies+[0]*(len(''.join(target_sequence[0:current_index[0]])+target_seq_rep[::-1])-len(target_indicies))
    for i in range(1,len(seq_relationship)+1):
        if seq_relationship[-i] == '|' or seq_relationship[-i] == '.':
            bound_indices.append(i+current_index[0])
    #print guide_indicies
    #print target_indicies
    return guide_seq_rep[::-1], seq_relationship[::-1], target_seq_rep[::-1], guide_indicies, target_indicies, current_index[0]
    


def define_features(guide_seq_rep, seq_relationship, target_seq_rep, relative_model_features):
    j = 0
    k = 0
    As = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    Cs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    Gs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    Us = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    guide_bulges = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    target_bulges = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #print guide_seq_rep
    #print seq_relationship
    #print target_seq_rep
    while j<21:
        if seq_relationship[k] == '|' or seq_relationship[k] == '.' or seq_relationship[k] == 'X':
            if target_seq_rep[k] == 'A':
                As[j] = 1
            elif target_seq_rep[k] == 'C':
                Cs[j] = 1
            elif target_seq_rep[k] == 'G':
                Gs[j] = 1
            elif target_seq_rep[k] == 'U':
                Us[j] = 1
            j = j+1
            k = k+1
        elif seq_relationship[k] == '-' and target_seq_rep[k] == '-':
            guide_bulges[j-1] +=1
            k+=1
            j+=1
        elif seq_relationship[k] == '-' and guide_seq_rep[k] == '-':
            target_bulges[j-1] +=1
            k+=1
        elif seq_relationship[k] == ' ' and guide_seq_rep[k] != ' ':
            if relative_model_features:
                if target_seq_rep[k] == 'A':
                    As[j] = 1
                elif target_seq_rep[k] == 'C':
                    Cs[j] = 1
                elif target_seq_rep[k] == 'G':
                    Gs[j] = 1
                elif target_seq_rep[k] == 'U':
                    Us[j] = 1
            j+=1
            k+=1
        if len(seq_relationship) == k:
            break
    print(As)
    print(Cs)
    print(Gs)
    print(Us)
    print(guide_bulges) # starting with g2 bulged out should be one less base possible
    print(target_bulges) # starting with 1/2 guide bulges
    return As+Cs+Gs+Us+guide_bulges+target_bulges


def replaceTsWithUs(originalList):
    '''function to replace all T's in a sequence with Us to convert from RNA to DNA'''
    for i in range(len(originalList)):
        originalList[i] = originalList[i].replace('T', 'U')
    return originalList

#########################################################################
#########################################################################
# main

# define params
guide = 'miR21'

if guide == "let7":
    guide_sequence = list('UGAUAUGUUGGAUGAUGGAGU')[::-1]
    wt_Target = 'ACUCCAUCAUCCAACAUAUCA'
else:
    # for miR21
    wt_Target = 'AUCGAAUAGUCUGACUACAAC'
    guide_sequence = list('UAGCUUAUCAGACUGAUGUUG')


relative_model_features = True
parvals = initialize_alignment_params(guide)
target_sequence = replaceTsWithUs(list('AAAAACAACATCAGTCTGATAAGCTAAAAAA'))[::-1]
MATCHED, MISMATCHED, TARGET_BULGE, GUIDE_BULGE, matched_min, matched_min_index = compute_score_matricies(target_sequence, guide_sequence, wt_Target, parvals)
#print(matched_min_index)
guide_seq_rep, seq_relationship, target_seq_rep, guide_indicies, target_indicies, current_index = traceback(MATCHED, MISMATCHED, TARGET_BULGE, GUIDE_BULGE, matched_min_index, target_sequence, guide_sequence, parvals)
feature_list = define_features(guide_seq_rep, seq_relationship, target_seq_rep, relative_model_features)


def align_sequence_list(sequences, guide, guide_sequence, wt_Target):
    parvals = initialize_alignment_params(guide)
    relative_model_features = True
    all_feature_lists = []
    target_indicies_list = []
    guide_indicies_list = []
    num_sequences = len(sequences)
    for k in range(num_sequences):
        target_sequence = replaceTsWithUs(list(sequences['Sequence'][k]))[::-1]
        MATCHED, MISMATCHED, TARGET_BULGE, GUIDE_BULGE, matched_min, matched_min_index = compute_score_matricies(target_sequence, guide_sequence, wt_Target, parvals)
        guide_seq_rep, seq_relationship, target_seq_rep, guide_indicies, target_indicies, index = traceback(MATCHED, MISMATCHED, TARGET_BULGE, GUIDE_BULGE, matched_min_index, target_sequence, guide_sequence, parvals)
        all_feature_lists.append(define_features(guide_seq_rep, seq_relationship, target_seq_rep, relative_model_features))
        guide_indicies_list.append(guide_indicies)
        target_indicies_list.append(target_indicies)
        print(sequences['Sequence'][k])
        print(str(k+1) + ' of ' + str(num_sequences) + '\n')
    sequences['features'] = all_feature_lists
    sequences['guide_indicies'] = guide_indicies_list
    sequences['target_indicies'] = target_indicies_list
    return sequences




