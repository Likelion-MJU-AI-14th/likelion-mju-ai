import random
import time

def show_status(name: str, weight: int, hp: int, hunger: int) -> None:
    print(f"이름: {name}")
    print(f"몸무게: {weight}")
    print(f"체력: {hp}")
    print(f"배고픔: {hunger}")
    
def create_character():
    print("=== 무인도 생존 시뮬레이션 ===")

    # 이름 입력 (문자형)
    name: str = input("당신의 이름은 무엇인가요? ")
    # 체중 설정 (숫자형 + random 모듈)
    weight = random.randint(45, 95)

    print(f"환영합니다, {name} 생존자님!")
    print(f"당신의 체중은 {weight}kg 으로 설정되었습니다.")

    # 초기 상태 (숫자형)
    hp = 80
    hunger = 86.7
    print(f"초기 상태 → 체력: {hp}, 배고픔: {hunger}")
    
    show_status(name=name, weight=weight, hp=hp, hunger=hunger)

create_character()

# 시간 지연 (time 모듈)
print("주위를 둘러보는 중...")
time.sleep(2)
print("아무것도 발견하지 못했다.")