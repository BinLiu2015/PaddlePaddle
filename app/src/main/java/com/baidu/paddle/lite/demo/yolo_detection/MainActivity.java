package com.baidu.paddle.lite.demo.yolo_detection;

import android.content.Intent;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentPagerAdapter;
import android.support.v4.view.ViewPager;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;

//import com.google.android.material.tabs.TabLayout;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //requestWindowFeature(Window.FEATURE_NO_TITLE);
        supportRequestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        this.setContentView(R.layout.first_act);

        Button camera = (Button) findViewById(R.id.camera);
        Button about = (Button) findViewById(R.id.about);
        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent(MainActivity.this, Demo.class);
                startActivity(i);
            }
        });

        about.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {
                setContentView(R.layout.about);

                ArrayList<String> arrayList = new ArrayList<>();
                arrayList.add("蚜虫（Aphid）");
                arrayList.add("斑点落叶病（Alternaria Spot）");
                arrayList.add("灰斑病（Gray Spot）");
                arrayList.add("花叶病（Mosaic）");
                arrayList.add("褐斑病（Brown Spot）");
                arrayList.add("铁锈病（Rust）");
            }
//                tabHost=(TabHost)findViewById(android.R.id.tabhost); //获取TabHost对象
//                tabHost.setup(); //初始化TabHost组件
//                //声明并实例化一个LayoutInflater对象
//                LayoutInflater inflater = LayoutInflater.from(this);
//                inflater.inflate(R.layout.disease_alternaria, tabHost.getTabContentView());
//                inflater.inflate(R.layout.disease_aphid, tabHost.getTabContentView());
//                inflater.inflate(R.layout.disease_brown, tabHost.getTabContentView());
//                inflater.inflate(R.layout.disease_gray, tabHost.getTabContentView());
//                inflater.inflate(R.layout.disease_rust, tabHost.getTabContentView());
//                inflater.inflate(R.layout.disease_mosaic, tabHost.getTabContentView());
//                tabHost.addTab(tabHost.newTabSpec("tab1").setIndicator("斑点落叶病").setContent(R.layout.disease_alternaria)); //添加第一个标签页
//                tabHost.addTab(tabHost.newTabSpec("tab2").setIndicator("蚜虫").setContent(R.layout.disease_aphid)); //添加第一个标签页
//                tabHost.addTab(tabHost.newTabSpec("tab3").setIndicator("褐斑病").setContent(R.layout.disease_brown)); //添加第一个标签页
//                tabHost.addTab(tabHost.newTabSpec("tab4").setIndicator("白粉病").setContent(R.layout.disease_gray)); //添加第一个标签页
//                tabHost.addTab(tabHost.newTabSpec("tab5").setIndicator("铁锈病").setContent(R.layout.disease_rust)); //添加第一个标签页
//                tabHost.addTab(tabHost.newTabSpec("tab5").setIndicator("花叶病").setContent(R.layout.disease_mosaic));

//                tabHost.setOnTabChangedListener(new TabHost.OnTabChangeListener() {
//                    @Override
//                    public void onTabChanged(String tabId) {//当切换选项按钮时调用
//                    }
//                });


        });
    }
}
