// Noam Koren 308192871
function memoize(fun){
  let cache = {}
  return function (n){
      if (cache[n] !== undefined ) {
          return cache[n]

      } else {
          let result = fun(n)
          cache[n] = result
          return result
      }
  }
}
function F(n) {
  if (n <= 1) 
    return n;
  else 
    return(F(n - 1) + F(n - 2));
}

console.log(F(80)) 
F = memoize(F)
console.log(F(80))