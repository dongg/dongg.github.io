---
layout: single
title: '01: github pages 사용법'
---

# 01: github pages 사용법

## github 프로파일 수정

* [깃허브 계정 제대로 꾸미기 (깃허브 프로파일 페이지 → 이력서로 만들기 팁🔥)](https://www.youtube.com/watch?v=w9DfC2BHGPA&ab_channel=%EB%93%9C%EB%A6%BC%EC%BD%94%EB%94%A9)
* 사용자이름과 같은 저장소를 생성: 
      * New repository: dongg/dongg
      * [v] Add README.md
* README.md 수정

## github pages 사용법
* [깃헙(GitHub) 블로그 10분안에 완성하기](https://www.youtube.com/watch?v=ACzFIAOsfpM&ab_channel=%ED%85%8C%EB%94%94%EB%85%B8%ED%8A%B8TeddyNote)


1. 테마선택
    * https://github.com/topics/jekyll-theme 에서 mmistakes / minimal-mistakes 선택 
2. mmistakes / minimal-mistakes의 우측 상단 [fork]하면 dongg/minimal-mistakes에 포크됨

3. [Settings]에서  dongg/minimal-mistakes
Repository name: dongg.github.io [Rename]

4. _config.yml  수정
    * url : "https://dongg.github.io"
    * 테마 변경
    * 수정후 Commit changes
    * 브라우저에서 dongg.github.io 로 확인 

5. 포스트 하기:
    * dongg.github.io  페이지에서 [Add file] [Create New File]
    * 모든 포스트는 _posts 폴더에 YYYY-MM-DD??????.md형식으로 작성해야 함
    * 포스트는 md이므로 제일 위에 YAML 헤더 포함 권장
      
```
---
layout: post
title: 포스트 01
---
```





