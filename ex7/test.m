a = [1 1; 3 3; 3 4; 5 8];
centroids = [0 0; 4 1; 3 0];
idx = zeros(size(a,1), 1);
data_rows = size(a,1);
K = size(centroids,1);
min_distance = 0;
for i = 1:data_rows
  cnt = 0;
  for j = 1:K
    c_x = (X(i) - centroids(j)).^2;
    c_y = (X(i + data_rows) - centroids(j + K)).^2;
    if(cnt == 0)
      min_distance = c_x + c_y;
      fprintf('d: %d\t', min_distance);
      idx(i) = j;
    else
      distance = c_x + c_y;
      fprintf('d: %d\t', distance);
      if(distance < min_distance)
        min_distance = distance;
        idx(i) = j;
      endif;
    endif;
    cnt ++;
   end;
   fprintf('min d: %d\n', min_distance);
end;
idx