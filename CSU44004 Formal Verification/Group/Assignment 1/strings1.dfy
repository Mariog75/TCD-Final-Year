method isPrefix(pre: string, str: string) returns (res:bool) 
    requires |pre| <= |str|
{
    if |pre| == 0
    {
        res := true;
    }
    var i := 0;
    var x : char;
    var y : char;
    var prefixLength := |pre|;
    while (i < prefixLength)
        decreases prefixLength - i;
        invariant 0 <= i <= prefixLength
    {  
        x := pre[i];
        y := str[i];

        if x != y 
        {
            res := false;
        }
        i := i + 1;  
    }  
    res := true; 
}

method isSubstring(sub: string, str: string) returns (res:bool)
    requires |sub| <= |str|
{
    var v := isPrefix(sub, str);
    if v
    {
        return true;
    }

    var i := 0;

    while i < |str|
        decreases |str| - i
    {
        if |sub| <= |str[i..]|
        {
            v := isPrefix(sub, str[i..]);
            if v 
            {
                return true;
            }
            
        }
        i := i + 1;
    }

    return false;
}
method haveCommonKSubstring(k: nat, str1: string, str2: string) returns (found: bool)
    requires k <= |str1| && k <= |str2|
    requires k > 0
{
    //get first k letters of str1
    //check isSubtring of str2
    //get next set of k letters 
    //check again etc.
    //terminate when k exceeds |str1|

    var x := 0;
    var subStr : string;
    var v : bool;
    var i : nat;

    
    while i <=(|str1| - k)
        decreases |str1| - k - i
    {

        subStr := str1[i..(k+i)];
        v := isSubstring(subStr, str2);

        if v {
            return true;
        }

        i := i+1;
    }
    return false;
    
}


method maxCommonSubstringLength(str1: string, str2: string) returns (len:nat)
{
    var x : nat;
    if |str1| <= |str2| 
    {
        x := |str1|;
    } else {
        x := |str2|;
    }

    if x < 1 {
        return 0;
    }

    var result := false;

    while x > 0
        decreases x - 0
    {
        result := haveCommonKSubstring(x, str1, str2);
        if result {
            return x;
        }
        x := x-1;
    }
    return 0;
}




